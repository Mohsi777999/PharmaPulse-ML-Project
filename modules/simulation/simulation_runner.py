"""
PharmaPulse - Simulation Runner
Top-level coordinator for the 1,095-day pharmacy simulation engine.
Imports modular subcomponents and reproduces full operational simulation.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from .demand_generator import generate_daily_demand
from .inventory_engine import process_expiries_for_medicine, allocate_fefo_sales
from .supplier_engine import evaluate_replenishment

def run_full_simulation(project_root=None, random_seed=42):
    """
    Executes the deterministic 3-year pharmacy simulation.
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    raw_dir = os.path.join(project_root, "data", "raw")

    random.seed(random_seed)
    np.random.seed(random_seed)

    df_medicines = pd.read_csv(os.path.join(raw_dir, "medicines.csv"))
    df_suppliers = pd.read_csv(os.path.join(raw_dir, "suppliers.csv"))
    suppliers_dict = {row["supplier_id"]: row.to_dict() for _, row in df_suppliers.iterrows()}

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2026, 12, 30)
    total_days = (end_date - start_date).days + 1
    dates = [start_date + timedelta(days=i) for i in range(total_days)]
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]

    batch_counter = 0
    po_counter = 0
    del_counter = 0
    mov_counter = 0

    all_batches = []
    all_pos = []
    all_deliveries = []
    all_daily_demand_sales = []
    all_daily_inventory = []
    all_movements = []

    active_batches_by_med = {m_id: [] for m_id in df_medicines["medicine_id"]}
    scheduled_deliveries = {d_str: [] for d_str in date_strs}

    med_params = {}
    for _, med in df_medicines.iterrows():
        m_id = med["medicine_id"]
        cat_id = med["category_id"]
        pattern = med["demand_pattern"]

        if cat_id in ["CAT001", "CAT005"]:
            base_demand = float(random.uniform(18.0, 32.0))
        elif cat_id in ["CAT002", "CAT003", "CAT004"]:
            base_demand = float(random.uniform(12.0, 22.0))
        else:
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

    # Day 1 opening batches
    for m_id, param in med_params.items():
        pref_sup = param["preferred_supplier_id"]
        num_init_batches = random.randint(1, 3)
        init_total_qty = int(param["reorder_level"] * random.uniform(1.1, 1.6))
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
            b_qty = max(15, b_qty)
            batch_counter += 1
            po_counter += 1
            del_counter += 1

            b_id = f"BATCH{batch_counter:06d}"
            po_id = f"PO{po_counter:06d}"
            del_id = f"DEL{del_counter:06d}"

            recv_d = datetime(2023, 12, 15) + timedelta(days=b_i * 4)
            exp_days = int(random.uniform(45, 120))
            exp_d = recv_d + timedelta(days=exp_days)

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

    pipeline_orders = {m_id: [] for m_id in med_params}

    for day_idx, curr_date in enumerate(dates):
        curr_date_str = date_strs[day_idx]
        dow = curr_date.weekday()
        month = curr_date.month

        # 1. Process deliveries
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

            for po_rec in all_pos:
                if po_rec["purchase_order_id"] == po_id:
                    po_rec["purchase_order_status"] = "DELIVERED"
                    po_rec["actual_delivery_date"] = curr_date_str
                    break

            pipeline_orders[m_id] = [p for p in pipeline_orders[m_id] if p["purchase_order_id"] != po_id]

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

        # 2. Daily operations per medicine
        for m_id, param in med_params.items():
            opening_stock = param["closing_stock"]

            # Expiries
            usable_batches, expired_qty, exp_movs, mov_counter = process_expiries_for_medicine(
                active_batches_by_med[m_id], curr_date_str, m_id, mov_counter
            )
            all_movements.extend(exp_movs)
            active_batches_by_med[m_id] = usable_batches

            received_qty = delivered_by_med[m_id]

            # Demand
            actual_demand = generate_daily_demand(
                param["base_demand"], param["demand_pattern"], param["category_id"],
                day_idx, total_days, dow, month
            )

            # Sales FEFO
            surviving_batches, units_sold, cogs, sale_movs, mov_counter = allocate_fefo_sales(
                active_batches_by_med[m_id], actual_demand, curr_date_str, m_id, mov_counter
            )
            active_batches_by_med[m_id] = surviving_batches
            all_movements.extend(sale_movs)

            unfulfilled_demand = actual_demand - units_sold
            revenue = round(units_sold * param["selling_price"], 2)
            cogs = round(cogs, 2)

            all_daily_demand_sales.append({
                "date": curr_date_str,
                "medicine_id": m_id,
                "actual_demand": actual_demand,
                "units_sold": units_sold,
                "unfulfilled_demand": unfulfilled_demand,
                "revenue": revenue,
                "cost_of_goods_sold": cogs
            })

            closing_stock = opening_stock + received_qty - expired_qty - units_sold
            param["closing_stock"] = closing_stock

            param["rolling_demand_window"].append(actual_demand)
            if len(param["rolling_demand_window"]) > 14:
                param["rolling_demand_window"].pop(0)
            avg_dem = sum(param["rolling_demand_window"]) / len(param["rolling_demand_window"])
            safe_denom = max(avg_dem, 0.1)
            days_of_inv = round(closing_stock / safe_denom, 2)
            stockout_flag = 1 if (closing_stock == 0 or unfulfilled_demand > 0) else 0

            all_daily_inventory.append({
                "date": curr_date_str,
                "medicine_id": m_id,
                "opening_stock": opening_stock,
                "received_quantity": received_qty,
                "expired_quantity": expired_qty,
                "units_sold": units_sold,
                "closing_stock": closing_stock,
                "stockout_flag": stockout_flag,
                "days_of_inventory_remaining": days_of_inv
            })

            # Replenishment
            p_qty = sum(p["ordered_quantity"] for p in pipeline_orders[m_id])
            replenish_res, po_counter = evaluate_replenishment(
                curr_date, curr_date_str, end_date, m_id, closing_stock, p_qty,
                param["reorder_level"], param["target_stock_days"], avg_dem,
                param["base_demand"], param["unit_cost"], param["preferred_supplier_id"],
                suppliers_dict, po_counter
            )
            if replenish_res is not None:
                po_record, delivery_task, o_qty = replenish_res
                all_pos.append(po_record)
                if delivery_task is not None:
                    arr_date = delivery_task["arrival_date_str"]
                    scheduled_deliveries[arr_date].append(delivery_task)
                pipeline_orders[m_id].append({
                    "purchase_order_id": po_record["purchase_order_id"],
                    "ordered_quantity": o_qty
                })

    return {
        "batches": pd.DataFrame(all_batches),
        "purchase_orders": pd.DataFrame(all_pos),
        "deliveries": pd.DataFrame(all_deliveries),
        "daily_demand_sales": pd.DataFrame(all_daily_demand_sales),
        "daily_inventory": pd.DataFrame(all_daily_inventory),
        "inventory_movements": pd.DataFrame(all_movements)
    }
