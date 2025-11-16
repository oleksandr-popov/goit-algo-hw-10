import pulp

# 1. Create the Linear Programming problem object
# We define the problem name and set the goal to be maximization (pulp.LpMaximize).
production_model = pulp.LpProblem("Beverage_Production_Optimization", pulp.LpMaximize)

# 2. Define the decision variables for the quantity of each drink
# 'lowBound=0' ensures production cannot be negative. 'cat="Continuous"' means
# the solution can be a fractional number (e.g., 15.5 units).
lemonade_qty = pulp.LpVariable(
    "Lemonade_Qty", lowBound=0, cat="Continuous"
)  # Quantity of Lemonade
juice_qty = pulp.LpVariable(
    "FruitJuice_Qty", lowBound=0, cat="Continuous"
)  # Quantity of Fruit Juice

# 3. Define the objective function - MAXIMIZE total products
# The goal is to maximize the sum of the quantities produced.
production_model += lemonade_qty + juice_qty, "Total_Products_to_Maximize"

# 4. Define the resource constraints (Limits on ingredients)
# The format is: model += (Constraint Expression) <= (Limit), "Constraint Name"

# Water constraint: 2 units for Lemonade, 1 unit for Juice. Max available: 100
production_model += 2 * lemonade_qty + 1 * juice_qty <= 100, "Water_Constraint"

# Sugar constraint: 1 unit for Lemonade. Max available: 50
production_model += 1 * lemonade_qty <= 50, "Sugar_Constraint"

# Lemon Juice constraint: 1 unit for Lemonade. Max available: 30
production_model += 1 * lemonade_qty <= 30, "LemonJuice_Constraint"

# Fruit Puree constraint: 2 units for Juice. Max available: 40
production_model += 2 * juice_qty <= 40, "FruitPuree_Constraint"

# 5. Solve the problem
production_model.solve()

# 6. Output the results
print("--- Linear Programming Optimization Results ---")

# Print the status of the solver (e.g., Optimal, Infeasible, Unbounded)
print(f"Solver Status: {pulp.LpStatus[production_model.status]}")

# Print the optimal quantity for each variable
print(f"Optimal Lemonade Quantity: {pulp.value(lemonade_qty)}")
print(f"Optimal Fruit Juice Quantity: {pulp.value(juice_qty)}")

# Print the value of the objective function at the optimal solution
print(f"Maximum Total Products: {pulp.value(production_model.objective)}")
