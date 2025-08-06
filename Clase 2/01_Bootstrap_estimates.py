import numpy as np

# Generar datos de muestra simulando números de serie de tanques capturados
np.random.seed(1234)  # Para reproducibilidad
sample_size = 5
true_N = 100
sample_data = np.random.randint(1, true_N + 1, size=sample_size)
print(f"Datos de la muestra: {sample_data}")

# Función para estimar el número total de tanques
def estimate_total_tanks(data):
    m = np.max(data)
    k = len(data)
    N_hat = m + (m / k) - 1
    return N_hat

# Estimación observada
observed_estimate = estimate_total_tanks(sample_data)
print(f"Estimación observada de N: {observed_estimate}")

# Generación de muestras bootstrap
n_bootstrap = 1000
bootstrap_estimates = np.array([estimate_total_tanks(np.random.choice(sample_data, size=len(sample_data), replace=True)) for _ in range(n_bootstrap)])

# Intervalo de confianza del 95%
ci_lower = np.percentile(bootstrap_estimates, 2.5)
ci_upper = np.percentile(bootstrap_estimates, 97.5)

print(f"Intervalo de confianza del 95%: [{ci_lower}, {ci_upper}]")
p_value = np.sum(bootstrap_estimates < 95) / n_bootstrap
print(f"P-valor: {p_value}")