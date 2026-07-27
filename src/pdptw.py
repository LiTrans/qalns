from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Node:
    idx: int
    request_id: int
    is_pickup: bool
    x: float
    y: float
    demand: int
    tw_open: float
    tw_close: float
    service: float = 5.0


@dataclass(frozen=True)
class Request:
    idx: int
    pickup_id: int
    delivery_id: int
    quantity: int


@dataclass(frozen=True)
class Vehicle:
    idx: int
    capacity: int
    max_route_time: float = 1e9


@dataclass
class PDPTWInstance:
    nodes: Dict[int, Node]
    requests: Dict[int, Request]
    vehicles: List[Vehicle]
    depot_x: float = 50.0
    depot_y: float = 50.0
    lateness_penalty: float = 50.0
    capacity_penalty: float = 250.0
    route_time_penalty: float = 20.0
    precedence_penalty: float = 1_000.0
    pairing_penalty: float = 2_000.0

    @property
    def request_ids(self) -> List[int]:
        return sorted(self.requests)

    @property
    def n_requests(self) -> int:
        return len(self.requests)

    @property
    def k_vehicles(self) -> int:
        return len(self.vehicles)

    def coord(self, node_id: int) -> Tuple[float, float]:
        if node_id == 0:
            return self.depot_x, self.depot_y
        node = self.nodes[node_id]
        return node.x, node.y

    def request_nodes(self, request_id: int) -> Tuple[int, int]:
        request = self.requests[request_id]
        return request.pickup_id, request.delivery_id


@dataclass
class RouteEval:
    travel: float
    lateness: float
    cap_violation: float
    route_time_violation: float
    precedence_violation: float
    objective: float
    feasible: bool


@dataclass
class SolutionEval:
    route_evals: List[RouteEval]
    objective: float
    feasible: bool
    missing_pairs: int
    duplicate_pairs: int


@dataclass
class RunTrace:
    method: str
    seed: int
    instance_tag: str
    iter_idx: List[int] = field(default_factory=list)
    best_objective: List[float] = field(default_factory=list)
    current_objective: List[float] = field(default_factory=list)
    elapsed_sec: List[float] = field(default_factory=list)
    quantum_calls: List[int] = field(default_factory=list)


def total_node_count(inst: PDPTWInstance) -> int:
    return 2 * inst.n_requests + 1


def fleet_capacity(inst: PDPTWInstance) -> int:
    return int(sum(vehicle.capacity for vehicle in inst.vehicles))


def total_demand(inst: PDPTWInstance) -> int:
    return int(sum(request.quantity for request in inst.requests.values()))


def capacity_utilization(inst: PDPTWInstance) -> float:
    cap = max(1, fleet_capacity(inst))
    return float(total_demand(inst) / cap)


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def deep_copy_routes(routes: List[List[int]]) -> List[List[int]]:
    return [list(route) for route in routes]


def objective_gap(value: float, reference: float) -> float:
    if abs(reference) < 1e-9:
        return 0.0
    return (value - reference) / abs(reference)


def evaluate_route(inst: PDPTWInstance, route: List[int], vehicle_idx: int) -> RouteEval:
    vehicle = inst.vehicles[vehicle_idx]
    travel = 0.0
    lateness = 0.0
    load = 0
    cap_violation = 0.0
    time_value = 0.0
    previous = 0
    seen_pickups: set[int] = set()
    precedence_violation = 0.0

    for node_id in route:
        distance = euclidean(inst.coord(previous), inst.coord(node_id))
        travel += distance
        time_value += distance

        node = inst.nodes[node_id]
        if time_value < node.tw_open:
            time_value = node.tw_open
        if time_value > node.tw_close:
            lateness += time_value - node.tw_close

        if node.is_pickup:
            seen_pickups.add(node.request_id)
        elif node.request_id not in seen_pickups:
            precedence_violation += 1.0

        load += node.demand
        cap_violation = max(cap_violation, max(0.0, load - vehicle.capacity))

        time_value += node.service
        previous = node_id

    back = euclidean(inst.coord(previous), inst.coord(0))
    travel += back
    time_value += back

    route_time_violation = max(0.0, time_value - vehicle.max_route_time)
    objective = (
        travel
        + inst.lateness_penalty * lateness
        + inst.capacity_penalty * cap_violation
        + inst.route_time_penalty * route_time_violation
        + inst.precedence_penalty * precedence_violation
    )
    feasible = (
        lateness <= 1e-9
        and cap_violation <= 1e-9
        and route_time_violation <= 1e-9
        and precedence_violation <= 1e-9
    )
    return RouteEval(
        travel=travel,
        lateness=lateness,
        cap_violation=cap_violation,
        route_time_violation=route_time_violation,
        precedence_violation=precedence_violation,
        objective=objective,
        feasible=feasible,
    )


def evaluate_solution(inst: PDPTWInstance, routes: List[List[int]], *, allow_partial: bool = False) -> SolutionEval:
    route_evals = [evaluate_route(inst, route, idx) for idx, route in enumerate(routes)]
    total_objective = sum(route_eval.objective for route_eval in route_evals)

    req_counts = defaultdict(int)
    req_vehicle: Dict[int, int] = {}
    pairing_violations = 0
    duplicate_pairs = 0

    for route_idx, route in enumerate(routes):
        route_req_counts = defaultdict(int)
        for node_id in route:
            route_req_counts[inst.nodes[node_id].request_id] += 1
        for req_id, count in route_req_counts.items():
            req_counts[req_id] += count
            if req_id in req_vehicle and req_vehicle[req_id] != route_idx:
                pairing_violations += 1
            req_vehicle[req_id] = route_idx

    missing_pairs = 0
    for req_id in inst.request_ids:
        count = req_counts.get(req_id, 0)
        if count == 0:
            missing_pairs += 1
        elif count != 2:
            pairing_violations += 1
            if count > 2:
                duplicate_pairs += 1

    if not allow_partial:
        total_objective += inst.pairing_penalty * (pairing_violations + missing_pairs)

    feasible = (
        all(route_eval.feasible for route_eval in route_evals)
        and pairing_violations == 0
        and (allow_partial or missing_pairs == 0)
    )
    return SolutionEval(route_evals, total_objective, feasible, missing_pairs, duplicate_pairs)


def generate_pdptw_instance(
    n_requests: int,
    k_vehicles: int,
    *,
    seed: int,
    tw_tightness: float = 0.5,
    capacity_slack: float = 0.30,
    pickup_delivery_span: float = 25.0,
    grid: float = 100.0,
) -> PDPTWInstance:
    rng = random.Random(seed)
    depot_x = grid / 2.0
    depot_y = grid / 2.0
    nodes: Dict[int, Node] = {}
    requests: Dict[int, Request] = {}
    total_quantity = 0
    base_width = max(18.0, 80.0 * (1.10 - 0.85 * tw_tightness))
    node_id = 1

    for req_id in range(1, n_requests + 1):
        quantity = rng.randint(1, 6)
        total_quantity += quantity

        pickup_x = rng.uniform(0, grid)
        pickup_y = rng.uniform(0, grid)
        delivery_x = min(grid, max(0.0, pickup_x + rng.uniform(-pickup_delivery_span, pickup_delivery_span)))
        delivery_y = min(grid, max(0.0, pickup_y + rng.uniform(-pickup_delivery_span, pickup_delivery_span)))

        travel_pd = euclidean((pickup_x, pickup_y), (delivery_x, delivery_y))
        center = euclidean((depot_x, depot_y), (pickup_x, pickup_y)) + rng.uniform(15.0, 90.0)
        pickup_width = base_width * rng.uniform(0.9, 1.2)
        delivery_width = base_width * rng.uniform(0.8, 1.1)

        pickup_open = max(0.0, center - pickup_width / 2.0)
        pickup_close = pickup_open + pickup_width
        delivery_open = max(pickup_open, center + 0.4 * travel_pd + rng.uniform(0.0, 18.0))
        delivery_close = delivery_open + delivery_width

        pickup_id = node_id
        delivery_id = node_id + 1
        node_id += 2

        nodes[pickup_id] = Node(
            idx=pickup_id,
            request_id=req_id,
            is_pickup=True,
            x=pickup_x,
            y=pickup_y,
            demand=quantity,
            tw_open=pickup_open,
            tw_close=pickup_close,
        )
        nodes[delivery_id] = Node(
            idx=delivery_id,
            request_id=req_id,
            is_pickup=False,
            x=delivery_x,
            y=delivery_y,
            demand=-quantity,
            tw_open=delivery_open,
            tw_close=delivery_close,
        )
        requests[req_id] = Request(req_id, pickup_id, delivery_id, quantity)

    avg_load = total_quantity / max(1, k_vehicles)
    vehicle_capacity = max(8, int(math.ceil(avg_load * (1.0 + capacity_slack))))
    vehicles = [Vehicle(idx=idx, capacity=vehicle_capacity, max_route_time=450.0) for idx in range(k_vehicles)]
    return PDPTWInstance(nodes=nodes, requests=requests, vehicles=vehicles, depot_x=depot_x, depot_y=depot_y)


def insert_request_into_route(
    inst: PDPTWInstance,
    base_route: List[int],
    request_id: int,
    pickup_slot: int,
    delivery_slot: int,
) -> List[int]:
    pickup_id, delivery_id = inst.request_nodes(request_id)
    route_len = len(base_route)
    out: List[int] = []
    for slot in range(route_len + 1):
        if slot == pickup_slot:
            out.append(pickup_id)
        if slot == delivery_slot:
            out.append(delivery_id)
        if slot < route_len:
            out.append(base_route[slot])
    return out


def best_pair_insertion(
    inst: PDPTWInstance,
    routes: List[List[int]],
    request_id: int,
) -> tuple[float, int, int, int, List[List[int]]]:
    current_eval = evaluate_solution(inst, routes)
    best_objective = float("inf")
    best_vehicle = 0
    best_pickup = 0
    best_delivery = 0
    best_routes = deep_copy_routes(routes)

    for vehicle_idx, route in enumerate(routes):
        for pickup_slot in range(len(route) + 1):
            for delivery_slot in range(pickup_slot, len(route) + 1):
                trial_routes = deep_copy_routes(routes)
                trial_routes[vehicle_idx] = insert_request_into_route(
                    inst,
                    route,
                    request_id,
                    pickup_slot,
                    delivery_slot,
                )
                objective = evaluate_solution(inst, trial_routes).objective
                if objective < best_objective:
                    best_objective = objective
                    best_vehicle = vehicle_idx
                    best_pickup = pickup_slot
                    best_delivery = delivery_slot
                    best_routes = trial_routes

    return best_objective - current_eval.objective, best_vehicle, best_pickup, best_delivery, best_routes


def greedy_initial_solution(inst: PDPTWInstance) -> List[List[int]]:
    routes = [[] for _ in range(inst.k_vehicles)]
    ordered_requests = sorted(
        inst.request_ids,
        key=lambda request_id: (
            inst.nodes[inst.requests[request_id].pickup_id].tw_close,
            inst.nodes[inst.requests[request_id].delivery_id].tw_close,
        ),
    )
    for request_id in ordered_requests:
        _, _, _, _, routes = best_pair_insertion(inst, routes, request_id)
    return routes


def route_load_slacks(inst: PDPTWInstance, routes: List[List[int]]) -> List[float]:
    slacks: List[float] = []
    for vehicle_idx, route in enumerate(routes):
        load = 0
        max_load = 0
        for node_id in route:
            load += inst.nodes[node_id].demand
            max_load = max(max_load, load)
        slacks.append(max(0.0, inst.vehicles[vehicle_idx].capacity - max_load))
    return slacks
