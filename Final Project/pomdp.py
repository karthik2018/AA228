# For gridworld, to keep it as simple as possible:
# 1. Food can only be in one corner at a time
# 2. Food is either present or not present
# 3. Food has a fixed nutritional (i.e. energy) value
# 4. If no corner has food, food grows in a random corner
# +------------+
# | A        B |
# |            |
# |            |
# |            |
# | C        D |
# +------------+
#
# For agent
# 1. Agent can move, look at a corner square to see if food is there, eat, or reproduce
# 2. Each turn agent loses energy from metabolism
# 3. As the agent gets old, it can fail to move when it tries
# 4. Reproducing resets age to 1, but has a energy cost
# 5. Agent dies (and simulation stops) if energy hits zero or agent reaches maximum age
#
# For reward
# 1. Agent gets a fixed bonus each turn it's alive
# 2. Agent gets a massive penalty for dying (running out of energy or getting too old)
# 3. Agent gets a fixed bonus for eating & reproducing (to encourage this behavior)
# 4. Agent gets a fixed penalty for moving (to encourage looking instead of wandering)
#
using QuickPOMDPs
using POMDPSimulators 
using ParticleFilters
using POMDPs
using POMDPModels # For the problem
using BasicPOMCP # For the solver
using POMDPPolicies # For creating a random policy
using Random # for AbstractRNG
using POMDPModelTools # for Deterministic
using QMDP

# Grid Definition
N = 5
A = (1, 1)
B = (1, N)
C = (N, 1)
D = (N, N)

FOOD_ENERGY = 10
ENERGY_COST_METABOLISM = 1  # cost per turn just for agent to exist
ENERGY_COST_MOVE = 1
ENERGY_COST_REPRODUCE = 3
AGENT_ENERGY_MAX = 15
AGENT_OLD_AGE = 10
AGENT_MAX_AGE = 20
DISCOUNT = 0.95

struct State
    food_a::Bool
    food_b::Bool
    food_c::Bool
    food_d::Bool
    agent_x::Int
    agent_y::Int
    agent_energy::Int
    agent_age::Int
end

MOVE = "move"
LOOK = "look"
EAT = "eat"
REPRODUCE = "reproduce"
struct Action
    type::String  # move, look, eat, nothing
    x::Int  # delta for move, absolute for look
    y::Int  # delta for move, absolute for look
end

struct Observation
    # These depend on what the action is. If it's a LOOK action, we find out
    # exactly what's in a corner.
    # XXX: Other actions get x,y of -1, -1. Clean up with custom type later.
    x::Int
    y::Int
    food::Bool

    # These are always fixed to the actual value of the state
    agent_x::Int
    agent_y::Int
    agent_energy::Int
    agent_age::Int
end

function all_states()
    states = []
    for x in 1:N
        for y in 1:N
            for energy in 0:AGENT_ENERGY_MAX
                for age in 0:AGENT_MAX_AGE
                    push!(states, State(0, 0, 0, 0, x, y, energy, age))
                    push!(states, State(1, 0, 0, 0, x, y, energy, age))
                    push!(states, State(0, 1, 0, 0, x, y, energy, age))
                    push!(states, State(0, 0, 1, 0, x, y, energy, age))
                    push!(states, State(0, 0, 0, 1, x, y, energy, age))
                end
            end
        end
    end
    return states
end

# Start with a square with food, so we don't die immediately (and sanity check we eat first)
STATES = all_states()
INITIAL_STATE = State(1, 0, 0, 0, 1, 1, 1, 0)
ACTIONS = [Action(MOVE,1,0), Action(MOVE,-1,0),
           Action(MOVE,0,1), Action(MOVE,0,-1),
           Action(MOVE,1,1), Action(MOVE,1,-1),
           Action(MOVE,-1,1), Action(MOVE,-1,-1),
           Action(LOOK,1,1), Action(LOOK,1,N),
           Action(LOOK,N,1), Action(LOOK,N,N),
           Action(EAT,-1,-1), Action(REPRODUCE,-1,-1)]

### Helper functions
function food_present(x::Int, y::Int, s::State)
    if (x, y) == A
        has_food = s.food_a
    elseif (x, y) == B
        has_food = s.food_b
    elseif (x, y) == C
        has_food = s.food_c
    elseif (x, y) == D
        has_food = s.food_d
    else
        return false
    end
end


### Main transition, observation, and reward logic
function get_next_state(s::State, a::Action)
    food_a = s.food_a
    food_b = s.food_b
    food_c = s.food_c
    food_d = s.food_d
    agent_x = s.agent_x
    agent_y = s.agent_y
    agent_energy = s.agent_energy
    agent_age = s.agent_age

    if a.type == MOVE
        if rand(0:s.agent_age) <= AGENT_OLD_AGE
            agent_x = clamp(agent_x + a.x, 1, N)
            agent_y = clamp(agent_y + a.y, 1, N)
        end
        agent_energy -= ENERGY_COST_MOVE
    elseif a.type == EAT
        if ((agent_x, agent_y) == A && food_a)
            food_a = false
            agent_energy += FOOD_ENERGY
        elseif ((agent_x, agent_y) == B && food_b)
            food_b = false
            agent_energy += FOOD_ENERGY
        elseif ((agent_x, agent_y) == C && food_c)
            food_c = false
            agent_energy += FOOD_ENERGY
        elseif ((agent_x, agent_y) == D && food_d)
            food_d = false
            agent_energy += FOOD_ENERGY
        end
    elseif a.type == REPRODUCE
        agent_age = 0
        agent_energy -= ENERGY_COST_REPRODUCE
    elseif a.type == LOOK
        # No change
    else
        throw(ArgumentError("Unrecognized action type"))
    end

    # Agent Metabolism
    agent_energy = clamp(agent_energy - ENERGY_COST_METABOLISM, 0, AGENT_ENERGY_MAX)

    # Agent aging
    agent_age = clamp(agent_age + 1, 0, AGENT_MAX_AGE)

    # Regrow food, if necessary
    if !any([food_a, food_b, food_c, food_d])
        r = randn()
        if r < 0.25
            food_a = true
        elseif r < 0.5
            food_b = true
        elseif r < 0.75
            food_c = true
        else
            food_d = true
        end
    end

    s = State(food_a, food_b, food_c, food_d, agent_x, agent_y, agent_energy, agent_age)
    return s
end


function get_observation(s::State, a::Action)
    if a.type == LOOK
        x = a.x
        y = a.y
        food = food_present(x, y, s)
    else
        x = -1
        y = -1
        food = false
    end

    return Observation(x, y, food, s.agent_x, s.agent_y, s.agent_age, s.agent_energy)
end

function get_reward(s::State, a::Action, sp::State)
    if sp.agent_energy <= 0 || sp.agent_age >= AGENT_MAX_AGE
        # Death
        return -1000000
    else
        # This should be all that's necessary, but we give some bonuses for life-like 
        # behavior in case solver can't see far enough ahead to avoid death case.
        reward = 1

        # Life-like behavior rewards
        if a.type == MOVE
            reward -= 1
        elseif a.type == REPRODUCE
            reward += 10
        elseif a.type == EAT
            reward += food_present(s.agent_x, s.agent_y, s) ? 10 : -10
        elseif a.type == LOOK
            # No special reward
        else
            throw(ArgumentError("Invalid Action Type"))
        end
    end
    return reward
end


### Definition of POMDP
struct LifePOMDP <: POMDP{State, Action, Observation}
end
function POMDPs.gen(m::LifePOMDP, s::State, a::Action, rng::AbstractRNG)
    # transition model
    sp = get_next_state(s, a)
    
    # observation model
    o = get_observation(s, a)
    
    # reward model
    r = get_reward(s, a, sp)
    
    # create and return a NamedTuple
    return (sp=sp, o=o, r=r)
end

function isterminal(s::State)
    return (s.agent_energy <= 0 || s.agent_age >= AGENT_MAX_AGE)
end

POMDPs.initialstate_distribution(m::LifePOMDP) = Deterministic(INITIAL_STATE)
POMDPs.actions(m::LifePOMDP) = ACTIONS
POMDPs.discount(m::LifePOMDP) = DISCOUNT
POMDPs.isterminal(m::LifePOMDP, s::State) = isterminal(s)

### Solution of POMDP
pomdp = LifePOMDP()
solver = POMCPSolver(max_depth=100)
policy = solve(solver, pomdp)

for (s, a, o, r, b) in stepthrough(pomdp, policy, "s,a,o,r,b", max_steps=100)
    @show s
    @show a
    @show o
    @show r
    #@show b
    println()
end
