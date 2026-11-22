"""
PharmaPulse - Simulation Supplier Engine
Manages vendor lead times, stochastic delivery delays, and replenishment purchase orders.
"""

import random
from datetime import timedelta

def evaluate_replenishment(
    curr_date,
    curr_date_str,
    end_date,
    medicine_id,
    closing_stock,
    pipeline_qty,
    reorder_level,
    target_stock_days,
    avg_recent_demand,
    base_demand,
    unit_cost,
    preferred_supplier_id,
    suppliers_dict,
    po_counter
):
    """
    Evaluates reorder conditions and creates future purchase orders.
    """
    effective_inventory = closing_stock + pipeline_qty

    if effective_inventory > reorder_level:
        return None, po_counter

    # Calculate required replenishment volume
    coverage_demand = max(avg_recent_demand, base_demand)
    target_qty = int(target_stock_days * coverage_demand)
    order_qty = max(30, target_qty - effective_inventory)

    sup_info = suppliers_dict[preferred_supplier_id]
    base_lead_time = int(sup_info["lead_time_days"])
    reliability = float(sup_info["delivery_reliability"])

    # Stochastic delay modeling
    delay_days = 0
    if random.random() > reliability:
        delay_days = int(random.choice([2, 4, 6, 8, 12]))

    actual_lead_time = base_lead_time + delay_days
    expected_del_date = curr_date + timedelta(days=base_lead_time)
    actual_del_date = curr_date + timedelta(days=actual_lead_time)

    po_counter += 1
    new_po_id = f"PO{po_counter:06d}"

    arrives_within_sim = actual_del_date <= end_date
    actual_del_str = actual_del_date.strftime("%Y-%m-%d") if arrives_within_sim else ""

    po_record = {
        "purchase_order_id": new_po_id,
        "medicine_id": medicine_id,
        "supplier_id": preferred_supplier_id,
        "order_date": curr_date_str,
        "ordered_quantity": order_qty,
        "expected_delivery_date": expected_del_date.strftime("%Y-%m-%d"),
        "actual_delivery_date": actual_del_str,
        "purchase_order_status": "PENDING",
        "unit_cost": unit_cost
    }

    delivery_task = None
    if arrives_within_sim:
        delivery_task = {
            "arrival_date_str": actual_del_str,
            "purchase_order_id": new_po_id,
            "medicine_id": medicine_id,
            "supplier_id": preferred_supplier_id,
            "delivered_quantity": order_qty,
            "unit_cost": unit_cost
        }

    return (po_record, delivery_task, order_qty), po_counter
