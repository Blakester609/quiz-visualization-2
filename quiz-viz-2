import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Set page layout
st.set_page_config(page_title="Gaussian Mixture Model", layout="wide")
st.title("2D Gaussian Mixture Visualization")

# Define coordinate grid
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)

def vectorized_gaussian_2d(X, Y, mu_x, mu_y, var_x, var_y):
    """Calculates the 2D Gaussian density using purely vectorized numpy operations."""
    norm = 1.0 / (2 * np.pi * np.sqrt(var_x * var_y))
    exponent = -0.5 * (((X - mu_x)**2) / var_x + ((Y - mu_y)**2) / var_y)
    return norm * np.exp(exponent)

# --- UI Setup ---
# Create tabs for clean organization
tab1, tab2, tab3 = st.tabs(["Gaussian 1", "Gaussian 2", "Gaussian 3"])

def create_sliders(key_prefix, default_w, default_mux, default_muy):
    """Helper to generate sliders for Streamlit."""
    w = st.slider("Weight", 0.0, 10.0, default_w, 0.1, key=f"{key_prefix}_w")
    col1, col2 = st.columns(2)
    with col1:
        mux = st.slider("Mean X", -8.0, 8.0, default_mux, 0.5, key=f"{key_prefix}_mux")
        varx = st.slider("Var X", 0.5, 10.0, 2.0, 0.5, key=f"{key_prefix}_varx")
    with col2:
        muy = st.slider("Mean Y", -8.0, 8.0, default_muy, 0.5, key=f"{key_prefix}_muy")
        vary = st.slider("Var Y", 0.5, 10.0, 2.0, 0.5, key=f"{key_prefix}_vary")
    return w, mux, muy, varx, vary

with tab1:
    w1, mux1, muy1, varx1, vary1 = create_sliders("g1", 1.0, -3.0, 3.0)
with tab2:
    w2, mux2, muy2, varx2, vary2 = create_sliders("g2", 1.0, 4.0, 2.0)
with tab3:
    w3, mux3, muy3, varx3, vary3 = create_sliders("g3", 1.0, -1.0, -4.0)

# Normalize weights
total_w = w1 + w2 + w3 + 1e-8 
w1, w2, w3 = w1 / total_w, w2 / total_w, w3 / total_w

# Compute densities
Z1 = vectorized_gaussian_2d(X, Y, mux1, muy1, varx1, vary1)
Z2 = vectorized_gaussian_2d(X, Y, mux2, muy2, varx2, vary2)
Z3 = vectorized_gaussian_2d(X, Y, mux3, muy3, varx3, vary3)
Z = w1 * Z1 + w2 * Z2 + w3 * Z3

# --- Plotting ---
fig = plt.figure(figsize=(14, 6))

# 2D Contour Plot
ax1 = fig.add_subplot(121)
ax1.contour(X, Y, Z, levels=25, cmap='viridis')
ax1.set_title('Contours of Constant Probability')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_xlim([-10, 10])
ax1.set_ylim([-10, 10])
ax1.grid(True, alpha=0.3)

# 3D Surface Plot
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.85, edgecolor='none')
ax2.set_title('Overall Mixture Density Surface')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Density p(x, y)')
ax2.set_xlim([-10, 10])
ax2.set_ylim([-10, 10])
ax2.set_zlim([0, 0.15]) 

plt.tight_layout()

# Render the plot in Streamlit
st.pyplot(fig)