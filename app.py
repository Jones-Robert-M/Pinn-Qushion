"""Pinn-Qushion: Interactive PINN Schrödinger Equation Solver."""

import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from pinn_qushion.inference import POTENTIAL_CONFIGS, ModelManager

st.set_page_config(
    page_title="Pinn-Qushion",
    page_icon="🌊",
    layout="wide",
)


@st.cache_resource
def get_model_manager():
    """Cache the model manager across reruns."""
    return ModelManager()


def potential_display_name_to_key(display_name: str) -> str:
    """Convert display name to internal key."""
    mapping = {
        "Infinite Square Well": "infinite_square_well",
        "Harmonic Oscillator": "harmonic_oscillator",
        "Finite Square Well": "finite_square_well",
        "Double Well": "double_well",
        "Gaussian Well": "gaussian_well",
    }
    return mapping.get(display_name, "infinite_square_well")


def main():
    st.title("Pinn-Qushion")
    st.subheader("Quantum Wavepacket Dynamics via Physics-Informed Neural Networks")

    # Initialize
    manager = get_model_manager()

    # Sidebar controls
    st.sidebar.header("Parameters")

    potential_display = st.sidebar.selectbox(
        "Potential Type",
        options=[
            "Infinite Square Well",
            "Harmonic Oscillator",
            "Finite Square Well",
            "Double Well",
            "Gaussian Well",
        ],
        index=0,
    )
    potential_key = potential_display_name_to_key(potential_display)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Initial Wavepacket")

    x0 = st.sidebar.slider(
        "Initial Position (x₀)",
        min_value=-5.0,
        max_value=5.0,
        value=0.0,
        step=0.1,
    )

    k0 = st.sidebar.slider(
        "Initial Momentum (k₀)",
        min_value=-3.0,
        max_value=3.0,
        value=1.0,
        step=0.1,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Time")

    t_value = st.sidebar.slider(
        "Time (t)",
        min_value=0.0,
        max_value=20.0,
        value=0.0,
        step=0.1,
    )

    # Compute wavefunction
    x = jnp.linspace(-10, 10, 256)
    psi_r, psi_i, prob = manager.predict(potential_key, x, t_value, x0, k0)

    # Get potential for overlay
    config = POTENTIAL_CONFIGS[potential_key]
    potential = config["class"](**config["params"])
    V = potential(x)

    # Normalize V for display (scale to fit with probability)
    V_display = np.array(V)
    V_max = np.max(np.abs(V_display[np.isfinite(V_display)]))
    if V_max > 0:
        V_display = V_display / V_max * np.max(prob) * 0.5

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Probability Density |Ψ(x,t)|²")

        fig = go.Figure()

        # Probability density
        fig.add_trace(go.Scatter(
            x=np.array(x),
            y=np.array(prob),
            mode="lines",
            name="|Ψ|²",
            line=dict(color="blue", width=2),
        ))

        # Potential overlay (scaled)
        fig.add_trace(go.Scatter(
            x=np.array(x),
            y=np.clip(V_display, -1, 1),
            mode="lines",
            name="V(x) (scaled)",
            line=dict(color="gray", width=1, dash="dash"),
        ))

        fig.update_layout(
            xaxis_title="Position (x)",
            yaxis_title="Probability Density",
            showlegend=True,
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Analysis")

        with st.expander("Energy Spectrum", expanded=True):
            st.info("Compute spectrum from time evolution (coming soon)")

        with st.expander("Wavefunction Components"):
            st.write(f"Max |Ψ|²: {float(jnp.max(prob)):.4f}")
            st.write(f"∫|Ψ|²dx: {float(jnp.sum(prob) * (x[1]-x[0])):.4f}")

    # Footer
    st.markdown("---")
    st.caption(
        "Built with JAX + Equinox + Streamlit | "
        "Pre-trained PINNs for quantum dynamics"
    )


if __name__ == "__main__":
    main()
