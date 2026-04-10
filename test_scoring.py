"""
Score validation script - checks if scores are mathematically achievable for all 3 tasks.
"""
from server.environment import SupplyChainEnv
from server.models import Action, Transfer

def test_task_1():
    env = SupplyChainEnv(task_id=1, seed=42)
    action = Action(transfers=[
        Transfer(source_node="central", destination_node="west", sku="SKU_A", quantity=50)
    ], advance_time=True)
    env.step(action)
    grade = env.grade_task_1()
    print(f"\n{'='*60}")
    print(f" TASK 1: Simple Restock (threshold=0.99)")
    print(f"{'='*60}")
    print(f"  transferred: {grade['transferred']}")
    print(f"  score: {grade['score']:.4f}")
    print(f"  result: {'PASS' if grade['score'] >= 0.99 else 'FAIL'}")
    return grade['score']

def test_task_2_do_nothing():
    env = SupplyChainEnv(task_id=2, seed=42)
    for _ in range(5):
        env.step(Action(transfers=[], advance_time=True))
    grade = env.grade_task_2()
    print(f"\n{'='*60}")
    print(f" TASK 2: Do Nothing (threshold=0.7)")
    print(f"{'='*60}")
    print(f"  stockout_days: {grade['stockout_days']}")
    print(f"  actual_cost: ${grade['actual_cost']:.2f}")
    print(f"  cost_excess: ${grade['cost_excess']:.2f}")
    print(f"  score: {grade['score']:.4f}")
    print(f"  result: {'PASS' if grade['score'] >= 0.7 else 'FAIL'} (should FAIL)")
    return grade['score']

def test_task_2_smart():
    env = SupplyChainEnv(task_id=2, seed=42)
    # Step 1: Transfer stock from central to east on Day 0
    action = Action(transfers=[
        Transfer(source_node="central", destination_node="east", sku="SKU_A", quantity=160),
        Transfer(source_node="central", destination_node="east", sku="SKU_B", quantity=120),
        Transfer(source_node="central", destination_node="east", sku="SKU_C", quantity=55),
        Transfer(source_node="central", destination_node="east", sku="SKU_E", quantity=50),
    ], advance_time=True)
    env.step(action)
    # Advance remaining 4 days
    for _ in range(4):
        env.step(Action(transfers=[], advance_time=True))
    grade = env.grade_task_2()
    print(f"\n{'='*60}")
    print(f" TASK 2: Smart Transfers (threshold=0.7)")
    print(f"{'='*60}")
    print(f"  stockout_days: {grade['stockout_days']}")
    print(f"  actual_cost: ${grade['actual_cost']:.2f}")
    print(f"  cost_excess: ${grade['cost_excess']:.2f}")
    print(f"  score: {grade['score']:.4f}")
    print(f"  result: {'PASS' if grade['score'] >= 0.7 else 'FAIL'} (should PASS)")
    return grade['score']

def test_task_2_rule_based():
    """Test using the same rule-based agent from inference.py"""
    from server.models import Observation
    env = SupplyChainEnv(task_id=2, seed=42)
    
    HORIZON = 5
    for day in range(HORIZON):
        obs = env.state()
        wh_inv = {wh.warehouse_id: dict(wh.current_inventory) for wh in obs.warehouses}
        wh_demand = {}
        counts = {}
        for f in obs.demand_forecast:
            wh_demand.setdefault(f.warehouse_id, {})
            counts.setdefault(f.warehouse_id, {})
            wh_demand[f.warehouse_id][f.sku] = wh_demand[f.warehouse_id].get(f.sku, 0.0) + f.quantity
            counts[f.warehouse_id][f.sku] = counts[f.warehouse_id].get(f.sku, 0) + 1
        for wh_id in wh_demand:
            for sku in wh_demand[wh_id]:
                n = counts[wh_id].get(sku, 1)
                wh_demand[wh_id][sku] /= n
        
        if obs.current_day > 0:
            env.step(Action(transfers=[], advance_time=True))
            continue
        
        days_remaining = max(1, HORIZON - obs.current_day)
        transfers = []
        nodes = list(wh_inv.keys())
        for sku in ["SKU_A", "SKU_B", "SKU_C", "SKU_D", "SKU_E"]:
            for dest in nodes:
                dest_inv = wh_inv[dest].get(sku, 0)
                dest_daily = wh_demand.get(dest, {}).get(sku, 0.0)
                if dest_daily <= 0: continue
                dest_needed = int(dest_daily * days_remaining)
                deficit = dest_needed - dest_inv
                if deficit <= 0: continue
                donor_candidates = [s for s in nodes if s != dest]
                donor_candidates.sort(key=lambda s: obs.shipping_rates.get(s, {}).get(dest, 999))
                for src in donor_candidates:
                    src_inv = wh_inv[src].get(sku, 0)
                    src_daily = wh_demand.get(src, {}).get(sku, 0.0)
                    src_needed = int(src_daily * days_remaining)
                    src_surplus = src_inv - src_needed
                    if src_surplus <= 0: continue
                    transfer_qty = min(deficit, src_surplus)
                    if transfer_qty <= 0: continue
                    transfers.append(Transfer(source_node=src, destination_node=dest, sku=sku, quantity=transfer_qty))
                    wh_inv[src][sku] = src_inv - transfer_qty
                    wh_inv[dest][sku] = dest_inv + transfer_qty
                    deficit -= transfer_qty
                    if deficit <= 0: break
        env.step(Action(transfers=transfers, advance_time=True))
    
    grade = env.grade_task_2()
    print(f"\n{'='*60}")
    print(f" TASK 2: Rule-Based Agent (threshold=0.7)")
    print(f"{'='*60}")
    print(f"  stockout_days: {grade['stockout_days']}")
    print(f"  actual_cost: ${grade['actual_cost']:.2f}")
    print(f"  cost_excess: ${grade['cost_excess']:.2f}")
    print(f"  score: {grade['score']:.4f}")
    print(f"  result: {'PASS' if grade['score'] >= 0.7 else 'FAIL'} (should PASS)")
    return grade['score']

def test_task_3_do_nothing():
    env = SupplyChainEnv(task_id=3, seed=42)
    for _ in range(7):
        env.step(Action(transfers=[], advance_time=True))
    grade = env.grade_task_3()
    print(f"\n{'='*60}")
    print(f" TASK 3: Do Nothing (threshold=0.6)")
    print(f"{'='*60}")
    print(f"  fulfillment_rate: {grade['fulfillment_rate']:.4f}")
    print(f"  total_fulfilled: {grade['total_fulfilled']}")
    print(f"  total_demanded: {grade['total_demanded']}")
    print(f"  total_costs: ${grade['total_costs']:.2f}")
    print(f"  score: {grade['score']:.4f}")
    print(f"  result: {'PASS' if grade['score'] >= 0.6 else 'FAIL'} (should FAIL or borderline)")
    return grade['score']

def test_task_3_smart():
    """Smart agent: transfer from low-demand central to high-demand east/west."""
    env = SupplyChainEnv(task_id=3, seed=42)
    
    HORIZON = 7
    for day in range(HORIZON):
        obs = env.state()
        wh_inv = {wh.warehouse_id: dict(wh.current_inventory) for wh in obs.warehouses}
        wh_demand = {}
        counts = {}
        for f in obs.demand_forecast:
            wh_demand.setdefault(f.warehouse_id, {})
            counts.setdefault(f.warehouse_id, {})
            wh_demand[f.warehouse_id][f.sku] = wh_demand[f.warehouse_id].get(f.sku, 0.0) + f.quantity
            counts[f.warehouse_id][f.sku] = counts[f.warehouse_id].get(f.sku, 0) + 1
        for wh_id in wh_demand:
            for sku in wh_demand[wh_id]:
                n = counts[wh_id].get(sku, 1)
                wh_demand[wh_id][sku] /= n
        
        days_remaining = max(1, HORIZON - obs.current_day)
        transfers = []
        nodes = list(wh_inv.keys())
        for sku in ["SKU_A", "SKU_B", "SKU_C", "SKU_D", "SKU_E"]:
            for dest in nodes:
                dest_inv = wh_inv[dest].get(sku, 0)
                dest_daily = wh_demand.get(dest, {}).get(sku, 0.0)
                if dest_daily <= 0: continue
                dest_needed = int(dest_daily * days_remaining)
                deficit = dest_needed - dest_inv
                if deficit <= 0: continue
                donor_candidates = [s for s in nodes if s != dest]
                donor_candidates.sort(key=lambda s: obs.shipping_rates.get(s, {}).get(dest, 999))
                for src in donor_candidates:
                    src_inv = wh_inv[src].get(sku, 0)
                    src_daily = wh_demand.get(src, {}).get(sku, 0.0)
                    src_needed = int(src_daily * days_remaining)
                    src_surplus = src_inv - src_needed
                    if src_surplus <= 0: continue
                    transfer_qty = min(deficit, src_surplus)
                    if transfer_qty <= 0: continue
                    transfers.append(Transfer(source_node=src, destination_node=dest, sku=sku, quantity=transfer_qty))
                    wh_inv[src][sku] = src_inv - transfer_qty
                    wh_inv[dest][sku] = dest_inv + transfer_qty
                    deficit -= transfer_qty
                    if deficit <= 0: break
        env.step(Action(transfers=transfers, advance_time=True))
    
    grade = env.grade_task_3()
    print(f"\n{'='*60}")
    print(f" TASK 3: Smart Transfers (threshold=0.6)")
    print(f"{'='*60}")
    print(f"  fulfillment_rate: {grade['fulfillment_rate']:.4f}")
    print(f"  total_fulfilled: {grade['total_fulfilled']}")
    print(f"  total_demanded: {grade['total_demanded']}")
    print(f"  total_costs: ${grade['total_costs']:.2f}")
    print(f"  score: {grade['score']:.4f}")
    print(f"  result: {'PASS' if grade['score'] >= 0.6 else 'FAIL'} (should PASS)")
    return grade['score']

# Run all
if __name__ == "__main__":
    print("\n" + "="*60)
    print(" SCORING VALIDATION")
    print("="*60)
    
    s1 = test_task_1()
    s2_nothing = test_task_2_do_nothing()
    s2_smart = test_task_2_smart()
    s2_rule = test_task_2_rule_based()
    s3_nothing = test_task_3_do_nothing()
    s3_smart = test_task_3_smart()
    
    print(f"\n{'='*60}")
    print(f" SUMMARY")
    print(f"{'='*60}")
    print(f"  Task 1 (transfer 50): {s1:.4f} {'PASS' if s1 >= 0.99 else 'FAIL'}")
    print(f"  Task 2 do-nothing:    {s2_nothing:.4f} {'PASS' if s2_nothing >= 0.7 else 'FAIL'}")
    print(f"  Task 2 smart:         {s2_smart:.4f} {'PASS' if s2_smart >= 0.7 else 'FAIL'}")
    print(f"  Task 2 rule-based:    {s2_rule:.4f} {'PASS' if s2_rule >= 0.7 else 'FAIL'}")
    print(f"  Task 3 do-nothing:    {s3_nothing:.4f} {'PASS' if s3_nothing >= 0.6 else 'FAIL'}")
    print(f"  Task 3 smart:         {s3_smart:.4f} {'PASS' if s3_smart >= 0.6 else 'FAIL'}")
