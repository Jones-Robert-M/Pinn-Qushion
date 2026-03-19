"""Pinn-Qushion: Interactive PINN Schrodinger Equation Solver."""

import time

import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from pinn_qushion.analysis import compute_autocorrelation, extract_spectrum
from pinn_qushion.inference import POTENTIAL_CONFIGS, ModelManager

st.set_page_config(
    page_title="Pinn-Qushion",
    page_icon="Q",
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


def compute_time_evolution(manager, potential_key, x, x0, k0, t_points):
    """Compute wavefunction at multiple time points for animation and analysis."""
    psi_r_list = []
    psi_i_list = []
    prob_list = []

    for t in t_points:
        psi_r, psi_i, prob = manager.predict(potential_key, x, float(t), x0, k0)
        psi_r_list.append(np.array(psi_r))
        psi_i_list.append(np.array(psi_i))
        prob_list.append(np.array(prob))

    return np.array(psi_r_list), np.array(psi_i_list), np.array(prob_list)


def main():
    st.title("Pinn-Qushion")
    st.markdown("**Quantum Wavepacket Dynamics via Physics-Informed Neural Networks**")

    # Initialize
    manager = get_model_manager()

    # Sidebar controls
    st.sidebar.header("Parameters")

    potential_display = st.sidebar.selectbox(
        "Potential Type",
        options=[
            "Harmonic Oscillator",
            "Infinite Square Well",
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
        "Initial Position (x0)",
        min_value=-5.0,
        max_value=5.0,
        value=0.0,
        step=0.1,
    )

    k0 = st.sidebar.slider(
        "Initial Momentum (k0)",
        min_value=-3.0,
        max_value=3.0,
        value=1.0,
        step=0.1,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Animation")

    # Animation controls
    animate = st.sidebar.checkbox("Enable Animation", value=False)

    if animate:
        speed = st.sidebar.slider(
            "Animation Speed",
            min_value=0.05,
            max_value=0.5,
            value=0.1,
            step=0.05,
        )
        t_max = st.sidebar.slider(
            "Max Time",
            min_value=5.0,
            max_value=20.0,
            value=10.0,
            step=1.0,
        )
    else:
        t_value = st.sidebar.slider(
            "Time (t)",
            min_value=0.0,
            max_value=20.0,
            value=0.0,
            step=0.1,
        )

    # Spatial grid
    x = jnp.linspace(-10, 10, 256)
    x_np = np.array(x)
    dx = float(x[1] - x[0])

    # Get potential for overlay
    config = POTENTIAL_CONFIGS[potential_key]
    potential = config["class"](**config["params"])
    V = np.array(potential(x))

    # Main content layout
    col_main, col_analysis = st.columns([2, 1])

    with col_main:
        st.subheader("Probability Density |Psi(x,t)|^2")
        plot_placeholder = st.empty()

    with col_analysis:
        st.subheader("Signal Analysis")
        analysis_placeholder = st.empty()

    # Compute time evolution for analysis
    n_time_points = 100
    t_points = np.linspace(0, 20, n_time_points)

    with st.spinner("Computing time evolution..."):
        psi_r_all, psi_i_all, prob_all = compute_time_evolution(
            manager, potential_key, x, x0, k0, t_points
        )

    # Compute autocorrelation and spectrum
    psi_0_r = psi_r_all[0]
    psi_0_i = psi_i_all[0]

    autocorr = []
    for i in range(len(t_points)):
        # C(t) = integral of psi*(x,0) * psi(x,t) dx
        c_real = np.sum(psi_0_r * psi_r_all[i] + psi_0_i * psi_i_all[i]) * dx
        c_imag = np.sum(psi_0_r * psi_i_all[i] - psi_0_i * psi_r_all[i]) * dx
        autocorr.append(np.sqrt(c_real**2 + c_imag**2))

    autocorr = np.array(autocorr)

    # FFT for spectrum
    dt = t_points[1] - t_points[0]
    spectrum = np.abs(np.fft.fft(autocorr))[:n_time_points // 2]
    freqs = np.fft.fftfreq(n_time_points, dt)[:n_time_points // 2]
    # Convert frequency to energy (E = hbar * omega, with hbar=1)
    energies = 2 * np.pi * freqs

    # Display analysis panel
    with analysis_placeholder.container():
        # Autocorrelation plot
        st.markdown("**Survival Probability |C(t)|**")
        fig_corr = go.Figure()
        fig_corr.add_trace(go.Scatter(
            x=t_points,
            y=autocorr,
            mode="lines",
            line=dict(color="green", width=2),
        ))
        fig_corr.update_layout(
            xaxis_title="Time",
            yaxis_title="|C(t)|",
            height=200,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # Spectrum plot
        st.markdown("**Energy Spectrum (FFT)**")
        fig_spec = go.Figure()
        fig_spec.add_trace(go.Scatter(
            x=energies[energies > 0],
            y=spectrum[energies > 0],
            mode="lines",
            line=dict(color="purple", width=2),
        ))
        fig_spec.update_layout(
            xaxis_title="Energy",
            yaxis_title="Amplitude",
            height=200,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_spec, use_container_width=True)

        # Statistics
        st.markdown("**Wavefunction Statistics**")
        norm_initial = float(np.sum(prob_all[0]) * dx)
        norm_final = float(np.sum(prob_all[-1]) * dx)
        st.text(f"Initial norm: {norm_initial:.4f}")
        st.text(f"Final norm:   {norm_final:.4f}")

    # Normalize V for display
    V_display = V.copy()
    V_finite = V_display[np.isfinite(V_display)]
    if len(V_finite) > 0:
        V_max = np.max(np.abs(V_finite))
        if V_max > 0:
            V_display = np.clip(V_display / V_max * 0.3, -0.5, 0.5)

    # Animation or static plot
    if animate:
        # Animation loop
        t_anim = np.linspace(0, t_max, int(t_max / speed))

        for t in t_anim:
            psi_r, psi_i, prob = manager.predict(potential_key, x, float(t), x0, k0)
            prob_np = np.array(prob)

            fig = go.Figure()

            # Potential overlay
            fig.add_trace(go.Scatter(
                x=x_np,
                y=V_display,
                mode="lines",
                name="V(x) scaled",
                line=dict(color="lightgray", width=1, dash="dash"),
                fill="tozeroy",
                fillcolor="rgba(200,200,200,0.2)",
            ))

            # Probability density
            fig.add_trace(go.Scatter(
                x=x_np,
                y=prob_np,
                mode="lines",
                name="|Psi|^2",
                line=dict(color="blue", width=2),
                fill="tozeroy",
                fillcolor="rgba(0,100,255,0.3)",
            ))

            fig.update_layout(
                xaxis_title="Position (x)",
                yaxis_title="Probability Density",
                yaxis_range=[0, max(1.0, float(np.max(prob_np)) * 1.2)],
                showlegend=True,
                height=400,
                title=f"t = {t:.2f}",
            )

            plot_placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(speed)

        st.rerun()
    else:
        # Static plot at selected time
        psi_r, psi_i, prob = manager.predict(potential_key, x, t_value, x0, k0)
        prob_np = np.array(prob)

        fig = go.Figure()

        # Potential overlay
        fig.add_trace(go.Scatter(
            x=x_np,
            y=V_display,
            mode="lines",
            name="V(x) scaled",
            line=dict(color="lightgray", width=1, dash="dash"),
            fill="tozeroy",
            fillcolor="rgba(200,200,200,0.2)",
        ))

        # Probability density
        fig.add_trace(go.Scatter(
            x=x_np,
            y=prob_np,
            mode="lines",
            name="|Psi|^2",
            line=dict(color="blue", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,100,255,0.3)",
        ))

        fig.update_layout(
            xaxis_title="Position (x)",
            yaxis_title="Probability Density",
            yaxis_range=[0, max(1.0, float(np.max(prob_np)) * 1.2)],
            showlegend=True,
            height=400,
            title=f"t = {t_value:.2f}",
        )

        plot_placeholder.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown("---")
    st.caption(
        "Built with JAX + Equinox + Streamlit | "
        "Physics-Informed Neural Networks for Quantum Dynamics"
    )


if __name__ == "__main__":
    main()
