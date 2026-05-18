"""A simple linear programming optimiser using hardcoded (fake) p50 Carbon intensity forecasts"""

from pulp import LpProblem, lpSum, LpMinimize, LpVariable, value

# Hardcoded forecast for testing
carbon_forecast: list[int] = [                                                                                                                                             
180, 185, 190, 175, 160, 150, 155, 170, 185, 190,  # 8am–10:30am                                                                                                       
175, 165, 155, 160, 170, 180, 185, 190, 195, 200   # 11am–6pm                                                                                                          
]  # 20 periods  

# Build the LP using this forecast
problem = LpProblem(name="minimize_total_carbon", sense=LpMinimize)
charge: dict = {}
for period in range(20):
    charge[period] = LpVariable(name=f"charge_{period}", lowBound=0, upBound=11)

# Objective: minimize total carbon 
problem += lpSum(vector=[carbon_forecast[i] * charge[i] for i in range(len(carbon_forecast))])


# Constraint: total energy >= 50 kWh (each period is 0.5 hours)                                                                                                            
problem += lpSum(vector=[charge[i] * 0.5 for i in range(20)]) >= 50, "min_energy_requirement"

# 5. Solve
problem.solve()

# 6. Extract results
optimal_schedule: dict = {period: charge[period].varValue for period in range(20)}
total_carbon = value(x=problem.objective)


print(optimal_schedule)