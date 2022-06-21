## Generative POMDP approach from Defining-a-POMDP-with-the-Generative-Interface.ipynb
## See https://github.com/JuliaPOMDP/POMDPExamples.jl/blob/master/notebooks/

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
# 1. Agent can move, eat, or reproduce
# 2. Each turn agent loses energy from metabolism
# 3. Reproducing resets age, but has a energy cost
# 4. Agent dies (and simulation stops) if energy hits zero or agent reaches maximum age
#
# For reward
# 1. Agent gets a fixed bonus each turn it's alive
# 2. Agent gets a massive penalty for dying (running out of energy or getting too old)
#
using QuickPOMDPs
using POMDPSimulators 
using POMDPs
using POMDPModels # For the problem
using BasicPOMCP # For the solver
using POMDPPolicies # For creating a random policy
using Random # for AbstractRNG
using POMDPModelTools # for Deterministic
using DiscreteValueIteration

# Grid Definition
N = 3
A = (1, 1)
B = (1, N)
C = (N, 1)
D = (N, N)

FOOD_ENERGY = 10
ENERGY_COST_METABOLISM = 1  # cost per turn just for agent to exist
ENERGY_COST_REPRODUCE = 3
AGENT_MAX_ENERGY = 10
AGENT_MAX_AGE = 7
DISCOUNT = 0.99

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
EAT = "eat"
REPRODUCE = "reproduce"
struct Action
    type::String  # move, eat, nothing
    x::Int  # delta for move
    y::Int  # delta for move
end

function all_states()::Vector{State}
    states::Vector{State} = []
    for x in 1:N
        for y in 1:N
            for energy in 0:AGENT_MAX_ENERGY
                for age in 0:AGENT_MAX_AGE
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
           Action(EAT,-1,-1),
           Action(REPRODUCE,-1,-1)]

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
function get_next_state(s::State, a::Action, sp::State)
    food_a = s.food_a
    food_b = s.food_b
    food_c = s.food_c
    food_d = s.food_d
    agent_x = s.agent_x
    agent_y = s.agent_y
    agent_energy = s.agent_energy
    agent_age = s.agent_age

    if a.type == MOVE
        agent_x = clamp(agent_x + a.x, 1, N)
        agent_y = clamp(agent_y + a.y, 1, N)
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
    else
        throw(ArgumentError("Unrecognized action type"))
    end

    # Agent Metabolism
    agent_energy = clamp(agent_energy - ENERGY_COST_METABOLISM, 0, AGENT_MAX_ENERGY)

    # Agent aging
    agent_age = clamp(agent_age + 1, 0, AGENT_MAX_AGE)

    # All values are deterministic besides food, which is deterministic unless it's all gone
    ns = State(food_a, food_b, food_c, food_d, agent_x, agent_y, agent_energy, agent_age)
    if !any([food_a, food_b, food_c, food_d])
        if (agent_x, agent_y, agent_energy, agent_age) == (sp.agent_x, sp.agent_y, sp.agent_energy, sp.agent_age)
            r = 0.25
        else
            r = 0
        end
    elseif sp == ns
        r = 1
    else
        r = 0
    end
    #println("get_next_state($s, $a, $sp): $ns, $r")
    return r
end

function get_reward(s::State, a::Action)
    if s.agent_energy <= 0 || s.agent_age >= AGENT_MAX_AGE
        # Death
        reward = -1000000
    else
        # Set the reward to be the energy level to encourage eating, discourage
        # moving, and discourage reproducing unless necessary to survive.
        reward = s.agent_energy
    end
    return reward
end

b0 = Uniform([INITIAL_STATE])
m = DiscreteExplicitMDP(all_states(), ACTIONS, get_next_state, get_reward, DISCOUNT, b0)

# creates the solver
solver = ValueIterationSolver(max_iterations=100, belres=1e-6, verbose=true) 
policy = solve(solver, m)

rsum = 0.0
for (s, a, r) in stepthrough(m, policy, "s,a,r", max_steps=10)
    @show s
    @show a
    @show r
    global rsum += r
    println("Current undiscounted reward total is $rsum")
    if s.agent_energy <= 0
        println("AGENT DIED OF STARVATION")
        break
    elseif s.agent_age >= AGENT_MAX_AGE
        println("AGENT DIED OF OLD AGE")
        break
    end
end
