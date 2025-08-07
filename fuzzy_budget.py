import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Input variables
income = ctrl.Antecedent(np.arange(0, 100001, 1000), 'income')
savings_goal = ctrl.Antecedent(np.arange(0, 50001, 1000), 'savings_goal')

# Output variable
recommended_saving = ctrl.Consequent(np.arange(0, 50001, 1000), 'recommended_saving')

# Membership functions
income.automf(3)
savings_goal.automf(3)

recommended_saving['low'] = fuzz.trimf(recommended_saving.universe, [0, 5000, 15000])
recommended_saving['medium'] = fuzz.trimf(recommended_saving.universe, [10000, 20000, 30000])
recommended_saving['high'] = fuzz.trimf(recommended_saving.universe, [25000, 40000, 50000])

# Rules
rule1 = ctrl.Rule(income['poor'] | savings_goal['poor'], recommended_saving['low'])
rule2 = ctrl.Rule(income['average'] & savings_goal['average'], recommended_saving['medium'])
rule3 = ctrl.Rule(income['good'] & savings_goal['good'], recommended_saving['high'])

# Control system
saving_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
saving_sim = ctrl.ControlSystemSimulation(saving_ctrl)

# Example usage
saving_sim.input['income'] = 60000
saving_sim.input['savings_goal'] = 25000
saving_sim.compute()

print(f"Recommended saving: ₹{saving_sim.output['recommended_saving']:.2f}")