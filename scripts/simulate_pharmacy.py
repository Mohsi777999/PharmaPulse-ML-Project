"""
PharmaPulse - Simulation Engine
Executes the 1,095-day deterministic daily simulation for 150 medicines across 10 suppliers.
Simulation Period: 2024-01-01 to 2026-12-30 (1,095 days).
Random Seed: 42.

Event order per day per medicine:
1. Opening inventory = previous closing inventory
2. Expiry processing (FEFO unavailable at start of expiry date)
3. Delivery arrivals (scheduled POs arriving today)
4. Demand generation (stochastic, independent of inventory)
5. FEFO sales allocation
6. Units sold & unfulfilled demand
7. Closing inventory calculation & balance check
8. Record daily inventory & inventory movements
9. Replenishment policy evaluation & PO generation
"""

import os
import math
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DATA_DIR = os.path.join(PROJECT_DIR, "data", "raw")

def run_simulation():
    print("Starting PharmaPulse 1,095-day Simulation Engine (Calibrated)...")
    
    # Load Master Data
    df_medicines = pd.read_csv(os.path.join(RAW_DATA_DIR, "medicines.csv"))
    df_suppliers = pd.read_csv(os.path.join(RAW_DATA_DIR, "suppliers.csv"))
    
    suppliers_dict = {row["supplier_id"]: row.to_dict() for _, row in df_suppliers.iterrows()}
    
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2026, 12, 30)
    total_days = (end_date - start_date).days + 1
    assert total_days == 1095, f"Expected 1095 days, got {total_days}"
    
    dates = [start_date + timedelta(days=i) for i in range(total_days)]
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]

    # Global tracking counters
    batch_counter = 0
    po_counter = 0
    del_counter = 0
    mov_counter = 0
    
    # Storage collections
    all_batches = []
    all_pos = []
    all_deliveries = []
    all_daily_demand_sales = []
    all_daily_inventory = []
    all_movements = []
    
    active_batches_by_med = {m_id: [] for m_id in df_medicines["medicine_id"]}
    scheduled_deliveries = {d_str: [] for d_str in date_strs}
    
    # Medicine simulation parameters
    med_params = {}
    for _, med in df_medicines.iterrows():
        m_id = med["medicine_id"]
        cat_id = med["category_id"]
        pattern = med["demand_pattern"]
        
        # Base daily demand calibrated by category
        if cat_id in ["CAT001", "CAT005"]:
            base_demand = float(random.uniform(18.0, 32.0))
        elif cat_id in ["CAT002", "CAT003", "CAT004"]:
            base_demand = float(random.uniform(12.0, 22.0))
        else: # CAT006 Vitamins
            base_demand = float(random.uniform(14.0, 24.0))
            
        med_params[m_id] = {
            "medicine_id": m_id,
            "category_id": cat_id,
            "demand_pattern": pattern,
            "base_demand": base_demand,
            "unit_cost": float(med["unit_cost"]),
            "selling_price": float(med["selling_price"]),
            "reorder_level": int(med["reorder_level"]),
            "target_stock_days": int(med["target_stock_days"]),
            "shelf_life_days": int(med["shelf_life_days"]),
            "preferred_supplier_id": med["preferred_supplier_id"],
            "closing_stock": 0,
            "rolling_demand_window": []
        }
        
    print("Initializing opening batches on Day 1 (2024-01-01)...")
    # Day 1: Create initial opening purchase orders and batches
    for m_id, param in med_params.items():
        pref_sup = param["preferred_supplier_id"]
        # Seed 1 to 3 initial batches
        num_init_batches = random.randint(1, 3)
        init_total_qty = int(param["reorder_level"] * random.uniform(1.1, 1.6))
        split_qtys = []
        if num_init_batches == 1:
            split_qtys = [init_total_qty]
        elif num_init_batches == 2:
            q1 = int(init_total_qty * random.uniform(0.4, 0.6))
            split_qtys = [q1, init_total_qty - q1]
        else:
            q1 = int(init_total_qty * random.uniform(0.25, 0.4))
            q2 = int((init_total_qty - q1) * random.uniform(0.4, 0.6))
            split_qtys = [q1, q2, init_total_qty - q1 - q2]
            
        opening_stock_sum = 0
        for b_i, b_qty in enumerate(split_qtys):
            if b_qty <= 0:
                b_qty = 15
            batch_counter += 1
            po_counter += 1
            del_counter += 1
            
            b_id = f"BATCH{batch_counter:06d}"
            po_id = f"PO{po_counter:06d}"
            del_id = f"DEL{del_counter:06d}"
            
            # Initial receipt dates in late Dec 2023
            recv_d = datetime(2023, 12, 15) + timedelta(days=b_i * 4)
            # Realistic synthetic shelf life (45 to 120 days after receipt)
            exp_days = int(random.uniform(45, 120))
            exp_d = recv_d + timedelta(days=exp_days)
            
            # Completed initial PO
            all_pos.append({
                "purchase_order_id": po_id,
                "medicine_id": m_id,
                "supplier_id": pref_sup,
                "order_date": (recv_d - timedelta(days=6)).strftime("%Y-%m-%d"),
                "ordered_quantity": b_qty,
                "expected_delivery_date": recv_d.strftime("%Y-%m-%d"),
                "actual_delivery_date": recv_d.strftime("%Y-%m-%d"),
                "purchase_order_status": "DELIVERED",
                "unit_cost": param["unit_cost"]
            })
            
            # Completed initial Delivery
            all_deliveries.append({
                "delivery_id": del_id,
                "purchase_order_id": po_id,
                "medicine_id": m_id,
                "supplier_id": pref_sup,
                "delivery_date": recv_d.strftime("%Y-%m-%d"),
                "delivered_quantity": b_qty,
                "delivery_status": "DELIVERED",
                "batch_id": b_id
            })
            
            batch_dict = {
                "batch_id": b_id,
                "medicine_id": m_id,
                "supplier_id": pref_sup,
                "purchase_order_id": po_id,
                "received_date": recv_d.strftime("%Y-%m-%d"),
                "expiry_date": exp_d.strftime("%Y-%m-%d"),
                "initial_quantity": b_qty,
                "remaining_quantity": b_qty,
                "sold_quantity": 0,
                "unsold_quantity_at_expiry": 0,
                "unit_cost": param["unit_cost"],
                "batch_status": "ACTIVE"
            }
            all_batches.append(batch_dict)
            active_batches_by_med[m_id].append(batch_dict)
            
            # Movement record
            mov_counter += 1
            all_movements.append({
                "movement_id": f"MOV{mov_counter:07d}",
                "date": "2024-01-01",
                "medicine_id": m_id,
                "batch_id": b_id,
                "movement_type": "opening_balance",
                "quantity": b_qty,
                "reference_id": po_id,
                "notes": "Initial inventory batch on simulation start"
            })
            opening_stock_sum += b_qty
            
        param["closing_stock"] = opening_stock_sum
        
    print(f"Initial setup complete: {batch_counter} batches seeded.")
    
    # Outstanding pipeline orders: medicine_id -> list of {po_id, ordered_quantity}
    pipeline_orders = {m_id: [] for m_id in med_params}
    
    # Main Daily Simulation Loop
    print("Simulating daily operations (1,095 days)...")
    for day_idx, curr_date in enumerate(dates):
        curr_date_str = date_strs[day_idx]
        dow = curr_date.weekday()
        month = curr_date.month
        
        # 1. Process Deliveries Arriving Today
        today_deliveries = scheduled_deliveries.get(curr_date_str, [])
        delivered_by_med = {m_id: 0 for m_id in med_params}
        for d_info in today_deliveries:
            m_id = d_info["medicine_id"]
            po_id = d_info["purchase_order_id"]
            sup_id = d_info["supplier_id"]
            qty = d_info["delivered_quantity"]
            cost = d_info["unit_cost"]
            
            batch_counter += 1
            del_counter += 1
            b_id = f"BATCH{batch_counter:06d}"
            del_id = f"DEL{del_counter:06d}"
            
            # Shelf life assumption: 45 to 150 days after receipt (per Section 4.E)
            # Irregular / seasonal end-of-season items have slight risk of expiry
            shelf_days = int(random.choice([50, 65, 80, 100, 120, 150]))
            exp_date = curr_date + timedelta(days=shelf_days)
            
            batch_dict = {
                "batch_id": b_id,
                "medicine_id": m_id,
                "supplier_id": sup_id,
                "purchase_order_id": po_id,
                "received_date": curr_date_str,
                "expiry_date": exp_date.strftime("%Y-%m-%d"),
                "initial_quantity": qty,
                "remaining_quantity": qty,
                "sold_quantity": 0,
                "unsold_quantity_at_expiry": 0,
                "unit_cost": cost,
                "batch_status": "ACTIVE"
            }
            all_batches.append(batch_dict)
            active_batches_by_med[m_id].append(batch_dict)
            
            # Delivery record
            all_deliveries.append({
                "delivery_id": del_id,
                "purchase_order_id": po_id,
                "medicine_id": m_id,
                "supplier_id": sup_id,
                "delivery_date": curr_date_str,
                "delivered_quantity": qty,
                "delivery_status": "DELIVERED",
                "batch_id": b_id
            })
            
            # Update PO status
            for po_rec in all_pos:
                if po_rec["purchase_order_id"] == po_id:
                    po_rec["purchase_order_status"] = "DELIVERED"
                    po_rec["actual_delivery_date"] = curr_date_str
                    break
                    
            # Remove from pipeline
            pipeline_orders[m_id] = [p for p in pipeline_orders[m_id] if p["purchase_order_id"] != po_id]
            
            # Movement record
            mov_counter += 1
            all_movements.append({
                "movement_id": f"MOV{mov_counter:07d}",
                "date": curr_date_str,
                "medicine_id": m_id,
                "batch_id": b_id,
                "movement_type": "purchase_receipt",
                "quantity": qty,
                "reference_id": po_id,
                "notes": f"Delivery received from {sup_id}"
            })
            
            delivered_by_med[m_id] += qty

        # Loop through each medicine for Day's events
        for m_id, param in med_params.items():
            opening_stock = param["closing_stock"]
            
            # 2. Process Expiries
            # A batch expiring on curr_date is unavailable starting at start-of-day curr_date
            expired_qty_today = 0
            usable_batches = []
            for b in active_batches_by_med[m_id]:
                if b["expiry_date"] <= curr_date_str:
                    rem = b["remaining_quantity"]
                    if rem > 0:
                        expired_qty_today += rem
                        b["unsold_quantity_at_expiry"] = rem
                        b["remaining_quantity"] = 0
                        mov_counter += 1
                        all_movements.append({
                            "movement_id": f"MOV{mov_counter:07d}",
                            "date": curr_date_str,
                            "medicine_id": m_id,
                            "batch_id": b["batch_id"],
                            "movement_type": "expiry",
                            "quantity": rem,
                            "reference_id": f"EXP_{curr_date_str}",
                            "notes": f"Batch expired on {b['expiry_date']}"
                        })
                    b["batch_status"] = "EXPIRED"
                else:
                    usable_batches.append(b)
            active_batches_by_med[m_id] = usable_batches
            
            # 3. Add Deliveries to usable stock
            received_qty_today = delivered_by_med[m_id]
            
            # 4. Generate Actual Demand (Independent of stock!)
            base = param["base_demand"]
            pattern = param["demand_pattern"]
            cat_id = param["category_id"]
            
            # Day of week factor
            dow_factor = 1.0 + 0.12 * math.sin(dow * math.pi / 3.0)
            
            # Seasonality factor
            season_factor = 1.0
            if pattern == "Seasonal":
                if cat_id in ["CAT001", "CAT003"]: # Fever / Cold peaks in winter (Dec - Feb)
                    season_factor = 1.0 + 0.50 * math.cos(2 * math.pi * (month - 1) / 12.0)
                elif cat_id in ["CAT004", "CAT006"]: # GI / Hydration peaks in summer (May - July)
                    season_factor = 1.0 + 0.45 * math.sin(2 * math.pi * (month - 3) / 12.0)
                else:
                    season_factor = 1.0 + 0.35 * math.sin(2 * math.pi * (month - 1) / 12.0)
            
            # Trend factor
            trend_factor = 1.0
            if pattern == "Growing":
                trend_factor = 1.0 + 0.70 * (day_idx / total_days)
            elif pattern == "Declining":
                trend_factor = max(0.35, 1.0 - 0.50 * (day_idx / total_days))
                
            # Irregular factor
            irregular_multiplier = 1.0
            if pattern == "Irregular":
                r_val = random.random()
                if r_val < 0.30:
                    irregular_multiplier = 0.1
                elif r_val > 0.88:
                    irregular_multiplier = float(random.uniform(2.5, 4.2))
                else:
                    irregular_multiplier = float(random.uniform(0.6, 1.4))
            
            # Occasional random outbreak/spike
            spike_factor = 1.0
            if random.random() < 0.025:
                spike_factor = float(random.uniform(1.8, 3.2))
                
            expected_demand = base * dow_factor * season_factor * trend_factor * irregular_multiplier * spike_factor
            expected_demand = max(0.1, expected_demand)
            
            # Generate stochastic count demand using Negative Binomial
            alpha = 0.20 if pattern != "Irregular" else 0.55
            mean_d = expected_demand
            var_d = mean_d + alpha * (mean_d ** 2)
            p_nb = mean_d / var_d
            n_nb = (mean_d ** 2) / (var_d - mean_d)
            actual_demand = int(np.random.negative_binomial(n=max(0.5, n_nb), p=min(0.99, max(0.01, p_nb))))
            if pattern == "Irregular" and irregular_multiplier < 0.2 and random.random() < 0.65:
                actual_demand = 0

            # 5. FEFO Sales Allocation
            active_batches_by_med[m_id].sort(key=lambda b: (b["expiry_date"], b["received_date"]))
            
            demand_remaining = actual_demand
            units_sold_today = 0
            cogs_today = 0.0
            
            surviving_batches = []
            for b in active_batches_by_med[m_id]:
                if demand_remaining <= 0:
                    surviving_batches.append(b)
                    continue
                    
                batch_rem = b["remaining_quantity"]
                if batch_rem <= 0:
                    continue
                    
                take_qty = min(demand_remaining, batch_rem)
                b["remaining_quantity"] -= take_qty
                b["sold_quantity"] += take_qty
                units_sold_today += take_qty
                cogs_today += take_qty * b["unit_cost"]
                demand_remaining -= take_qty
                
                mov_counter += 1
                all_movements.append({
                    "movement_id": f"MOV{mov_counter:07d}",
                    "date": curr_date_str,
                    "medicine_id": m_id,
                    "batch_id": b["batch_id"],
                    "movement_type": "sale",
                    "quantity": take_qty,
                    "reference_id": f"SALE_{curr_date_str}_{m_id}",
                    "notes": f"FEFO sale from batch {b['batch_id']}"
                })
                
                if b["remaining_quantity"] > 0:
                    surviving_batches.append(b)
                else:
                    b["batch_status"] = "DEPLETED"
                    
            active_batches_by_med[m_id] = surviving_batches
            
            # 6. Sales, Unfulfilled Demand & Revenue
            unfulfilled_demand = actual_demand - units_sold_today
            revenue = round(units_sold_today * param["selling_price"], 2)
            cogs_today = round(cogs_today, 2)
            
            all_daily_demand_sales.append({
                "date": curr_date_str,
                "medicine_id": m_id,
                "actual_demand": actual_demand,
                "units_sold": units_sold_today,
                "unfulfilled_demand": unfulfilled_demand,
                "revenue": revenue,
                "cost_of_goods_sold": cogs_today
            })
            
            # 7. Closing Inventory Calculation & Reconciliation
            closing_stock = opening_stock + received_qty_today - expired_qty_today - units_sold_today
            assert closing_stock >= 0, f"Negative closing stock {closing_stock} for {m_id} on {curr_date_str}"
            batch_stock_check = sum(b["remaining_quantity"] for b in active_batches_by_med[m_id])
            assert closing_stock == batch_stock_check, f"Reconciliation error on {curr_date_str}: closing {closing_stock} != batches sum {batch_stock_check}"
            
            param["closing_stock"] = closing_stock
            
            # Rolling demand window for days of inventory
            param["rolling_demand_window"].append(actual_demand)
            if len(param["rolling_demand_window"]) > 14:
                param["rolling_demand_window"].pop(0)
            avg_recent_demand = sum(param["rolling_demand_window"]) / len(param["rolling_demand_window"])
            safe_demand_denom = max(avg_recent_demand, 0.1)
            days_of_inventory = round(closing_stock / safe_demand_denom, 2)
            
            stockout_flag = 1 if (closing_stock == 0 or unfulfilled_demand > 0) else 0
            
            all_daily_inventory.append({
                "date": curr_date_str,
                "medicine_id": m_id,
                "opening_stock": opening_stock,
                "received_quantity": received_qty_today,
                "expired_quantity": expired_qty_today,
                "units_sold": units_sold_today,
                "closing_stock": closing_stock,
                "stockout_flag": stockout_flag,
                "days_of_inventory_remaining": days_of_inventory
            })
            
            # 8. Replenishment Policy Evaluation & Future PO Generation
            pipeline_qty = sum(p["ordered_quantity"] for p in pipeline_orders[m_id])
            effective_inventory = closing_stock + pipeline_qty
            
            if effective_inventory <= param["reorder_level"]:
                coverage_demand = max(avg_recent_demand, param["base_demand"])
                target_qty = int(param["target_stock_days"] * coverage_demand)
                order_qty = max(30, target_qty - effective_inventory)
                
                pref_sup_id = param["preferred_supplier_id"]
                sup_info = suppliers_dict[pref_sup_id]
                base_lead_time = int(sup_info["lead_time_days"])
                reliability = float(sup_info["delivery_reliability"])
                
                delay_days = 0
                if random.random() > reliability:
                    delay_days = int(random.choice([2, 4, 6, 8, 12]))
                    
                actual_lead_time = base_lead_time + delay_days
                expected_del_date = curr_date + timedelta(days=base_lead_time)
                actual_del_date = curr_date + timedelta(days=actual_lead_time)
                
                po_counter += 1
                new_po_id = f"PO{po_counter:06d}"
                
                if actual_del_date <= end_date:
                    po_status = "PENDING"
                    actual_del_str = actual_del_date.strftime("%Y-%m-%d")
                    scheduled_deliveries[actual_del_str].append({
                        "purchase_order_id": new_po_id,
                        "medicine_id": m_id,
                        "supplier_id": pref_sup_id,
                        "delivered_quantity": order_qty,
                        "unit_cost": param["unit_cost"]
                    })
                else:
                    po_status = "PENDING"
                    actual_del_str = ""
                    
                all_pos.append({
                    "purchase_order_id": new_po_id,
                    "medicine_id": m_id,
                    "supplier_id": pref_sup_id,
                    "order_date": curr_date_str,
                    "ordered_quantity": order_qty,
                    "expected_delivery_date": expected_del_date.strftime("%Y-%m-%d"),
                    "actual_delivery_date": actual_del_str,
                    "purchase_order_status": po_status,
                    "unit_cost": param["unit_cost"]
                })
                
                pipeline_orders[m_id].append({
                    "purchase_order_id": new_po_id,
                    "ordered_quantity": order_qty
                })

        if (day_idx + 1) % 150 == 0 or (day_idx + 1) == total_days:
            print(f"Progress: Day {day_idx + 1}/{total_days} ({curr_date_str}) processed...")

    print("Simulation complete! Saving raw CSV files...")
    
    # Write batches.csv
    df_batches = pd.DataFrame(all_batches)
    df_batches = df_batches[[
        "batch_id", "medicine_id", "supplier_id", "purchase_order_id",
        "received_date", "expiry_date", "initial_quantity", "unit_cost", "batch_status"
    ]]
    df_batches.to_csv(os.path.join(RAW_DATA_DIR, "batches.csv"), index=False)
    print(f"batches.csv saved: {len(df_batches)} rows")
    
    # Write purchase_orders.csv
    df_pos = pd.DataFrame(all_pos)
    df_pos = df_pos[[
        "purchase_order_id", "medicine_id", "supplier_id", "order_date",
        "ordered_quantity", "expected_delivery_date", "actual_delivery_date",
        "purchase_order_status", "unit_cost"
    ]]
    df_pos.to_csv(os.path.join(RAW_DATA_DIR, "purchase_orders.csv"), index=False)
    print(f"purchase_orders.csv saved: {len(df_pos)} rows")
    
    # Write deliveries.csv
    df_del = pd.DataFrame(all_deliveries)
    df_del = df_del[[
        "delivery_id", "purchase_order_id", "medicine_id", "supplier_id",
        "delivery_date", "delivered_quantity", "delivery_status", "batch_id"
    ]]
    df_del.to_csv(os.path.join(RAW_DATA_DIR, "deliveries.csv"), index=False)
    print(f"deliveries.csv saved: {len(df_del)} rows")
    
    # Write daily_demand_sales.csv
    df_sales = pd.DataFrame(all_daily_demand_sales)
    df_sales = df_sales[[
        "date", "medicine_id", "actual_demand", "units_sold",
        "unfulfilled_demand", "revenue", "cost_of_goods_sold"
    ]]
    df_sales.to_csv(os.path.join(RAW_DATA_DIR, "daily_demand_sales.csv"), index=False)
    print(f"daily_demand_sales.csv saved: {len(df_sales)} rows")
    
    # Write daily_inventory.csv
    df_inv = pd.DataFrame(all_daily_inventory)
    df_inv = df_inv[[
        "date", "medicine_id", "opening_stock", "received_quantity",
        "expired_quantity", "units_sold", "closing_stock",
        "stockout_flag", "days_of_inventory_remaining"
    ]]
    df_inv.to_csv(os.path.join(RAW_DATA_DIR, "daily_inventory.csv"), index=False)
    print(f"daily_inventory.csv saved: {len(df_inv)} rows")
    
    # Write inventory_movements.csv
    df_mov = pd.DataFrame(all_movements)
    df_mov = df_mov[[
        "movement_id", "date", "medicine_id", "batch_id",
        "movement_type", "quantity", "reference_id", "notes"
    ]]
    df_mov.to_csv(os.path.join(RAW_DATA_DIR, "inventory_movements.csv"), index=False)
    print(f"inventory_movements.csv saved: {len(df_mov)} rows")
    
    print("All raw simulation CSVs successfully generated and verified!")

if __name__ == "__main__":
    run_simulation()
