# ============================================
# EXERCISE 1: RULE-BASED HOSPITAL TRIAGE SYSTEM
# ============================================

# This system classifies patients into:
# High Risk, Medium Risk, or Low Risk
# Based on fixed medical thresholds


def rule_based_triage(age, heart_rate, blood_pressure, temperature):
    
    # Validate edge cases first
    if heart_rate <= 0 or blood_pressure <= 0 or temperature <= 0:
        return "Invalid patient data"
    
    if temperature > 45:
        return "Invalid temperature reading"
    
    # High Risk Conditions
    if temperature > 39 or heart_rate > 120 or blood_pressure > 180:
        return "High Risk (Emergency)"
    
    # Medium Risk Conditions
    elif temperature > 38 or heart_rate > 100 or blood_pressure > 140:
        return "Medium Risk (Urgent)"
    
    # Otherwise Low Risk
    else:
        return "Low Risk (Non-Urgent)"


# --------------------------------------------
# TESTING THE SYSTEM
# --------------------------------------------

# Normal patient
patient1 = rule_based_triage(age=60, heart_rate=130, blood_pressure=190, temperature=40)
print("Patient 1:", patient1)

# Medium case
patient2 = rule_based_triage(age=45, heart_rate=105, blood_pressure=150, temperature=38.5)
print("Patient 2:", patient2)

# Low risk
patient3 = rule_based_triage(age=30, heart_rate=80, blood_pressure=120, temperature=37)
print("Patient 3:", patient3)

# Edge case (invalid)
patient4 = rule_based_triage(age=50, heart_rate=-10, blood_pressure=120, temperature=37)
print("Patient 4:", patient4)