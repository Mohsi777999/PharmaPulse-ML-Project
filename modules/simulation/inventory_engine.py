"""
PharmaPulse - Simulation Inventory Engine
Enforces FEFO (First Expiry, First Out) sales allocation, daily batch expirations,
and physical inventory conservation laws.
"""

def process_expiries_for_medicine(active_batches, curr_date_str, medicine_id, mov_counter):
    """
    Identifies batches expiring on or before curr_date_str.
    Expired units are removed from usable stock before demand is served.
    """
    expired_qty = 0
    usable_batches = []
    movements = []

    for b in active_batches:
        if b["expiry_date"] <= curr_date_str:
            rem = b["remaining_quantity"]
            if rem > 0:
                expired_qty += rem
                b["unsold_quantity_at_expiry"] = rem
                b["remaining_quantity"] = 0
                mov_counter += 1
                movements.append({
                    "movement_id": f"MOV{mov_counter:07d}",
                    "date": curr_date_str,
                    "medicine_id": medicine_id,
                    "batch_id": b["batch_id"],
                    "movement_type": "expiry",
                    "quantity": rem,
                    "reference_id": f"EXP_{curr_date_str}",
                    "notes": f"Batch expired on {b['expiry_date']}"
                })
            b["batch_status"] = "EXPIRED"
        else:
            usable_batches.append(b)

    return usable_batches, expired_qty, movements, mov_counter

def allocate_fefo_sales(usable_batches, actual_demand, curr_date_str, medicine_id, mov_counter):
    """
    Fulfills customer demand using FEFO: Earliest expiry batch consumed first.
    Returns:
    surviving_batches, units_sold, cogs, movements, updated_mov_counter
    """
    # Sort usable batches by expiry_date ASC, received_date ASC
    usable_batches.sort(key=lambda b: (b["expiry_date"], b["received_date"]))

    demand_remaining = actual_demand
    units_sold = 0
    cogs = 0.0
    movements = []
    surviving_batches = []

    for b in usable_batches:
        if demand_remaining <= 0:
            surviving_batches.append(b)
            continue

        batch_rem = b["remaining_quantity"]
        if batch_rem <= 0:
            continue

        take_qty = min(demand_remaining, batch_rem)
        b["remaining_quantity"] -= take_qty
        b["sold_quantity"] += take_qty
        units_sold += take_qty
        cogs += take_qty * b["unit_cost"]
        demand_remaining -= take_qty

        mov_counter += 1
        movements.append({
            "movement_id": f"MOV{mov_counter:07d}",
            "date": curr_date_str,
            "medicine_id": medicine_id,
            "batch_id": b["batch_id"],
            "movement_type": "sale",
            "quantity": take_qty,
            "reference_id": f"SALE_{curr_date_str}_{medicine_id}",
            "notes": f"FEFO sale from batch {b['batch_id']}"
        })

        if b["remaining_quantity"] > 0:
            surviving_batches.append(b)
        else:
            b["batch_status"] = "DEPLETED"

    return surviving_batches, units_sold, cogs, movements, mov_counter
