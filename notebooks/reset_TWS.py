#!/usr/bin/env python3
"""
Simple script to connect to TWS and close all positions.
This script will:
1. Connect to TWS
2. Get all portfolio positions
3. Close all positions (buy if negative, sell if positive)
"""

import time
from ib_insync import IB, Stock, util, MarketOrder

def main():
    # Initialize IB connection
    ib = IB()
    
    try:
        # Connect to TWS (default localhost:7497 for live, 7496 for paper)
        # Change to 7496 if using paper trading
        print("Connecting to TWS...")
        ib.connect('127.0.0.1', 7497, clientId=1)
        print("Connected successfully!")
        
        # Get portfolio positions
        print("Getting portfolio positions...")
        positions = ib.positions()
        
        if not positions:
            print("No positions found in portfolio.")
            return
        
        print(f"Found {len(positions)} positions:")
        
        # Process each position
        for position in positions:
            contract = position.contract
            current_position = position.position
            
            print(f"\nProcessing: {contract.symbol} - Current position: {current_position}")
            
            if current_position == 0:
                print(f"  {contract.symbol}: Position is already 0, skipping")
                continue
            
            # Create stock contract with SMART routing
            stock = Stock(contract.symbol, 'SMART', contract.currency)
            
            # Qualify the contract to ensure proper routing
            ib.qualifyContracts(stock)
            
            # Determine action and quantity
            if current_position > 0:
                # We have a long position, need to sell
                action = 'SELL'
                quantity = abs(current_position)
                print(f"  {contract.symbol}: Selling {quantity} shares to close long position")
            else:
                # We have a short position, need to buy
                action = 'BUY'
                quantity = abs(current_position)
                print(f"  {contract.symbol}: Buying {quantity} shares to close short position")
            
            # Create and place SMART market order
            order = MarketOrder(action, quantity, transmit=True)
            
            # Place the order with SMART routing
            trade = ib.placeOrder(stock, order)
            print(f"  SMART order placed for {contract.symbol}: {action} {quantity} shares")
            
            # Monitor order status
            print(f"  Order status: {trade.orderStatus.status}")
            
            # Wait and check order execution
            for i in range(15):  # Wait up to 15 seconds
                if trade.orderStatus.status in ['Filled', 'Cancelled']:
                    break
                print(f"  Order status update: {trade.orderStatus.status}")
            
            if trade.orderStatus.status == 'Filled':
                print(f"  ✓ Order filled for {contract.symbol}")
                if hasattr(trade, 'fills') and trade.fills:
                    fill = trade.fills[-1]
                    print(f"    Fill price: ${fill.execution.price}, Time: {fill.time}")
            else:
                print(f"  ⚠ Order status: {trade.orderStatus.status}")
            
            # Wait a bit between orders
        
        print("\nAll closing orders have been placed!")
        print("Waiting 5 seconds for orders to process...")
        
        # Check final positions
        print("\nFinal portfolio check:")
        final_positions = ib.positions()
        
        if not final_positions:
            print("Portfolio is now empty - all positions closed successfully!")
        else:
            print("Remaining positions:")
            for pos in final_positions:
                if pos.position != 0:
                    print(f"  {pos.contract.symbol}: {pos.position}")
    
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure TWS is running and API connections are enabled")
        print("In TWS: File > Global Configuration > API > Settings > Enable ActiveX and Socket Clients")
    
    finally:
        # Disconnect
        if ib.isConnected():
            ib.disconnect()
            print("\nDisconnected from TWS")

if __name__ == "__main__":
    main()