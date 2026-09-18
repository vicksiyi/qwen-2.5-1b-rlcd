"""Bounded forecasts using real moving-body rules, not a static obstacle grid."""
from collections import Counter, deque

VECTORS = ((0, -1), (1, 0), (0, 1), (-1, 0))
LOOKAHEAD = 6


def step(body, direction, food, size):
    dx, dy = VECTORS[direction]
    head = (body[0][0] + dx, body[0][1] + dy)
    eats = head == food
    remaining = body if eats else body[:-1]
    if not (0 <= head[0] < size and 0 <= head[1] < size) or head in remaining:
        return None
    return (head,) + remaining, eats


def region(body, size):
    """Optimistic static connectivity: allow the tail that will vacate next tick."""
    blocked = set(body[1:-1])
    distances = {body[0]: 0}
    queue = deque([body[0]])
    while queue:
        x, y = queue.popleft()
        for dx, dy in VECTORS:
            point = (x + dx, y + dy)
            if (0 <= point[0] < size and 0 <= point[1] < size
                    and point not in blocked and point not in distances):
                distances[point] = distances[(x, y)] + 1
                queue.append(point)
    return len(distances), body[-1] in distances


def survival_depth(body, direction, food, size, depth=LOOKAHEAD):
    """Maximum feasible horizon up to depth, with exact growth/tail movement."""
    if depth == 0 or len(body) == size * size:
        return depth
    best = 0
    for turn in (0, -1, 1):
        facing = (direction + turn) % 4
        advanced = step(body, facing, food, size)
        if advanced is None:
            continue
        nxt, eats = advanced
        reach = 1 + survival_depth(nxt, facing, None if eats else food, size, depth - 1)
        best = max(best, reach)
        if best == depth:
            break
    return best


def escape_status(body, direction, size):
    if len(body) == size * size:
        return True
    _, tail_connected = region(body, size)
    return tail_connected and survival_depth(body, direction, None, size) == LOOKAHEAD


def static_route(body, food, size):
    blocked = set(body[1:-1])
    queue = deque([body[0]])
    parents = {body[0]: None}
    while queue:
        point = queue.popleft()
        if point == food:
            moves = []
            while parents[point] is not None:
                point, direction = parents[point]
                moves.append(direction)
            return moves[::-1]
        for direction, (dx, dy) in enumerate(VECTORS):
            nxt = (point[0] + dx, point[1] + dy)
            if (0 <= nxt[0] < size and 0 <= nxt[1] < size
                    and nxt not in blocked and nxt not in parents):
                parents[nxt] = (point, direction)
                queue.append(nxt)
    return None


def food_forecast(body, direction, food, size):
    """Find a validated route; bounded beam fallback may miss existing routes.

    Return (steps, post-food escape estimate). None means no route FOUND, not
    proof of unreachability. Future randomly spawned food is never assumed.
    """
    route = static_route(body, food, size)
    fallback = (None, None)
    if route is not None:
        current, facing = body, direction
        for move in route:
            if move == (facing + 2) % 4:
                break
            advanced = step(current, move, food, size)
            if advanced is None:
                break
            current, eats = advanced
            facing = move
        else:
            if current[0] == food:
                safe = escape_status(current, facing, size)
                if safe:
                    return len(route), True
                fallback = (len(route), False)
    # Search over complete snake configurations; the tail really moves.
    frontier = [(body, direction)]
    seen = {(body, direction)}
    for distance in range(1, min(40, size * 2 + 8) + 1):
        expanded = []
        for current, facing in frontier:
            for turn in (0, -1, 1):
                move = (facing + turn) % 4
                advanced = step(current, move, food, size)
                if advanced is None:
                    continue
                nxt, eats = advanced
                if eats:
                    if escape_status(nxt, move, size):
                        return distance, True
                    if fallback[0] is None or distance < fallback[0]:
                        fallback = (distance, False)
                    continue
                key = (nxt, move)
                if key in seen:
                    continue
                seen.add(key)
                expanded.append(key)
        if not expanded:
            break
        expanded.sort(key=lambda item: abs(item[0][0][0] - food[0]) + abs(item[0][0][1] - food[1]))
        frontier = expanded[:24]
    return fallback


def enrich(state, observations):
    body = tuple(tuple(p) for p in state.snake)
    food = tuple(state.food)
    visits = Counter(tuple(p) for p in state.recent_heads)
    for item in observations:
        advanced = step(body, item['direction'], food, state.size)
        item.update(recent_visits=0, survival_steps=0, tail_connected=False,
                    route_steps=None, escape_after_food=None, space_sufficient=False)
        if advanced is None:
            continue
        nxt, eats = advanced
        space, tail = region(nxt, state.size)
        horizon = 1 + survival_depth(nxt, item['direction'], None if eats else food, state.size, LOOKAHEAD - 1)
        if eats:
            route_steps, escape = 1, escape_status(nxt, item['direction'], state.size)
        else:
            distance, escape = food_forecast(nxt, item['direction'], food, state.size)
            route_steps = distance + 1 if distance is not None else None
        item.update(recent_visits=visits[nxt[0]], survival_steps=horizon,
                    tail_connected=tail, route_steps=route_steps,
                    escape_after_food=escape, space_sufficient=space >= len(nxt),
                    reachable_cells=space)
    return observations


def planning_candidates(observations, steps_since_food, size):
    """Explicit optional hybrid assistance. Never rewrites model probabilities."""
    safe = [o for o in observations if o['safe']]
    if not safe:
        return [], 'no_safe_move'
    depth = max(o['survival_steps'] for o in safe)
    viable = [o for o in safe if o['survival_steps'] == depth]
    with_escape = [o for o in viable if o['tail_connected'] and o['space_sufficient']]
    if with_escape:
        viable = with_escape
    routes = [o for o in viable if o['route_steps'] is not None and o['escape_after_food']]
    if routes:
        # Model breaks ties between equally short routes with an escape estimate.
        shortest = min(o['route_steps'] for o in routes)
        shortest_routes = [o for o in routes if o['route_steps'] == shortest]
        fewest = min(o['recent_visits'] for o in shortest_routes)
        return [o for o in shortest_routes if o['recent_visits'] == fewest], 'safe_food_route'
    if steps_since_food >= size * 2:
        fewest = min(o['recent_visits'] for o in viable)
        return [o for o in viable if o['recent_visits'] == fewest], 'break_loop'
    return viable, 'survival_forecast'
