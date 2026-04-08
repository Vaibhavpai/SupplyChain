"""
tests/test_models.py

One instantiation test per Pydantic v2 model.
Run with:  pytest tests/test_models.py -v
"""

import pytest
from pydantic import ValidationError

from models import (
    Action,
    DemandForecast,
    IncomingShipment,
    Observation,
    ProcessSupervisionReward,
    RewardBreakdown,
    StepResult,
    Transfer,
    WarehouseState,
)


# ---------------------------------------------------------------------------
# IncomingShipment
# ---------------------------------------------------------------------------

def test_incoming_shipment_valid():
    s = IncomingShipment(
        source_node="supplier_1",
        sku="SKU_A",
        quantity=50,
        arrives_on_day=3,
    )
    assert s.source_node == "supplier_1"
    assert s.quantity == 50


def test_incoming_shipment_rejects_float_quantity():
    with pytest.raises(ValidationError):
        IncomingShipment(
            source_node="supplier_1",
            sku="SKU_A",
            quantity=1.5,      # must be int
            arrives_on_day=3,
        )


# ---------------------------------------------------------------------------
# WarehouseState
# ---------------------------------------------------------------------------

def test_warehouse_state_valid():
    ws = WarehouseState(
        warehouse_id="east",
        current_inventory={"SKU_A": 100, "SKU_B": 50},
        incoming_shipments=[],
    )
    assert ws.warehouse_id == "east"
    assert ws.current_inventory["SKU_A"] == 100


def test_warehouse_state_with_shipments():
    shipment = IncomingShipment(
        source_node="central", sku="SKU_C", quantity=20, arrives_on_day=2
    )
    ws = WarehouseState(
        warehouse_id="west",
        current_inventory={"SKU_C": 10},
        incoming_shipments=[shipment],
    )
    assert len(ws.incoming_shipments) == 1


# ---------------------------------------------------------------------------
# DemandForecast
# ---------------------------------------------------------------------------

def test_demand_forecast_valid():
    df = DemandForecast(warehouse_id="central", sku="SKU_B", day=5, quantity=30)
    assert df.day == 5
    assert df.quantity == 30


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

def _make_observation(day: int = 0) -> Observation:
    ws = WarehouseState(
        warehouse_id="east",
        current_inventory={"SKU_A": 10},
        incoming_shipments=[],
    )
    df = DemandForecast(warehouse_id="east", sku="SKU_A", day=day + 1, quantity=5)
    return Observation(
        current_day=day,
        warehouses=[ws],
        shipping_rates={"east": {"central": 2.0}},
        transit_times={"east": {"central": 1}},
        demand_forecast=[df],
        active_alerts=[],
    )


def test_observation_valid():
    obs = _make_observation(day=0)
    assert obs.current_day == 0
    assert len(obs.warehouses) == 1
    assert obs.shipping_rates["east"]["central"] == 2.0


def test_observation_model_json_schema():
    schema = Observation.model_json_schema()
    assert "properties" in schema
    assert "current_day" in schema["properties"]


# ---------------------------------------------------------------------------
# Transfer
# ---------------------------------------------------------------------------

def test_transfer_valid():
    t = Transfer(
        source_node="east",
        destination_node="central",
        sku="SKU_A",
        quantity=10,
    )
    assert t.quantity == 10


def test_transfer_rejects_missing_field():
    with pytest.raises(ValidationError):
        Transfer(source_node="east", destination_node="central", sku="SKU_A")
        # quantity missing


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

def test_action_default_empty_transfers():
    a = Action()
    assert a.transfers == []
    assert a.advance_time is True


def test_action_with_transfers():
    t = Transfer(source_node="west", destination_node="east", sku="SKU_D", quantity=5)
    a = Action(transfers=[t], advance_time=False)
    assert len(a.transfers) == 1
    assert a.advance_time is False


def test_action_model_json_schema():
    schema = Action.model_json_schema()
    assert "properties" in schema
    assert "transfers" in schema["properties"]


# ---------------------------------------------------------------------------
# RewardBreakdown
# ---------------------------------------------------------------------------

def test_reward_breakdown_valid():
    r = RewardBreakdown(
        fulfilled_demand_reward=100.0,
        holding_cost_penalty=12.5,
        shipping_cost_penalty=20.0,
        invalid_action_penalty=0.0,
        total=67.5,
    )
    assert r.total == 67.5


def test_reward_breakdown_rejects_missing_field():
    with pytest.raises(ValidationError):
        RewardBreakdown(
            fulfilled_demand_reward=10.0,
            holding_cost_penalty=1.0,
            # shipping_cost_penalty missing
            invalid_action_penalty=0.0,
            total=9.0,
        )


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------

def test_step_result_valid():
    obs = _make_observation(day=1)
    reward = ProcessSupervisionReward(
        fulfilled_demand_reward=50.0,
        stockout_penalty=0.0,
        holding_cost_penalty=-5.0,
        shipping_cost_penalty=-10.0,
        inventory_balance_reward=0.0,
        demand_forecast_accuracy_reward=0.0,
        proactive_restocking_bonus=0.0,
        safety_margin_bonus=0.0,
        temporal_incentive_bonus=0.0,
        invalid_action_penalty=0.0,
        safety_violation_penalty=0.0,
        total=35.0,
    )
    result = StepResult(observation=obs, reward=reward, done=False, info={})
    assert result.done is False
    assert result.reward.total == 35.0