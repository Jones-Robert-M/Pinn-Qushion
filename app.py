"""Pinn-Qushion: Interactive PINN Schrodinger Equation Solver."""

import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from pinn_qushion.analysis import compute_autocorrelation_series, compute_energy_spectrum
from pinn_qushion.inference import POTENTIAL_CONFIGS, ModelManager

st.set_page_config(
    page_title="Pinn-Qushion",
    page_icon="Q",
    layout="wide",
)

POTENTIAL_DISPLAY_NAMES = {
    "Harmonic Oscillator": "harmonic_oscillator",
    "Infinite Square Well": "infinite_square_well",
    "Finite Square Well": "finite_square_well",
    "Double Well": "double_well",
    "Gaussian Well": "gaussian_well",
}

POTENTIAL_PARAMS_DISPLAY = {
    "harmonic_oscillator": "ω = 1.0",
    "infinite_square_well": "width = 8.0",
    "finite_square_well": "width = 6.0, depth = 5.0",
    "double_well": "separation = 4.0, depth = 5.0, barrier = 3.0",
    "gaussian_well": "depth = 5.0, σ = 2.0",
}

POTENTIAL_ANALYTIC_ENERGIES = {
    "harmonic_oscillator": {
        "formula": "Eₙ = ω(n + ½)",
        "levels": [0.5, 1.5, 2.5, 3.5, 4.5],
    },
    "infinite_square_well": {
        "formula": "Eₙ = n²π²/(2L²), L=4",
        "levels": [round((n**2 * np.pi**2) / (2 * 4.0**2), 3) for n in range(1, 6)],
    },
    "finite_square_well": None,
    "double_well": None,
    "gaussian_well": None,
}


@st.cache_resource
def get_model_manager() -> ModelManager:
    return ModelManager()


@st.cache_data
def compute_evolution(potential_key: str, x0: float, k0: float):
    """Pre-compute full time evolution for animation and analysis.

    Cached on (potential_key, x0, k0) — recomputes only when parameters change.
    Returns numpy arrays ready for Plotly.
    """
    manager = get_model_manager()
    x = np.linspace(-10, 10, 256)
    dx = x[1] - x[0]
    t_points = np.linspace(0, 20, 120)

    x_jnp = jnp.array(x)
    n = len(x_jnp)
    x0_arr = jnp.full(n, x0)
    k0_arr = jnp.full(n, k0)

    prob_frames = np.zeros((len(t_points), n))
    psi_r_series = np.zeros((len(t_points), n))
    psi_i_series = np.zeros((len(t_points), n))

    for i, t in enumerate(t_points):
        t_arr = jnp.full(n, float(t))
        psi_r, psi_i = manager.get_model(potential_key).psi(x_jnp, t_arr, x0_arr, k0_arr)
        psi_r_series[i] = np.array(psi_r)
        psi_i_series[i] = np.array(psi_i)
        prob_frames[i] = psi_r_series[i] ** 2 + psi_i_series[i] ** 2

    # Complex wavefunction series for autocorrelation
    psi_complex = psi_r_series + 1j * psi_i_series
    psi_0 = jnp.array(psi_complex[0])
    psi_series_jnp = jnp.array(psi_complex)
    C_t = compute_autocorrelation_series(psi_0, psi_series_jnp, float(dx))
    C_t_np = np.array(C_t)

    dt = float(t_points[1] - t_points[0])
    energies, amplitudes = compute_energy_spectrum(C_t, dt)

    norms = np.sum(prob_frames, axis=1) * float(dx)

    # ⟨x⟩(t) = ∫ x |Ψ|² dx
    x_mean = np.sum(prob_frames * x[np.newaxis, :], axis=1) * float(dx)

    # ⟨p⟩(t) = ∫ (Ψ_R ∂Ψ_I/∂x − Ψ_I ∂Ψ_R/∂x) dx
    dpsi_r_dx = np.gradient(psi_r_series, float(dx), axis=1)
    dpsi_i_dx = np.gradient(psi_i_series, float(dx), axis=1)
    p_mean = np.sum(
        psi_r_series * dpsi_i_dx - psi_i_series * dpsi_r_dx, axis=1
    ) * float(dx)

    config = POTENTIAL_CONFIGS[potential_key]
    potential = config["class"](**config["params"])
    V = np.array(potential(x_jnp))

    return {
        "x": x,
        "t_points": t_points,
        "prob_frames": prob_frames,
        "norms": norms,
        "C_t_abs": np.abs(C_t_np),
        "energies": np.array(energies),
        "amplitudes": np.array(amplitudes),
        "x_mean": x_mean,
        "p_mean": p_mean,
        "V": V,
        "dx": float(dx),
    }


def build_animation(data: dict, potential_key: str) -> go.Figure:
    """Build a Plotly figure with pre-computed animation frames."""
    x = data["x"]
    t_points = data["t_points"]
    prob_frames = data["prob_frames"]
    V = data["V"]

    V_finite = V[np.isfinite(V)]
    V_abs_max = np.max(np.abs(V_finite)) if len(V_finite) > 0 else 0.0
    V_scale = V_abs_max if V_abs_max > 0 else 1.0
    prob_max = float(np.max(prob_frames)) if np.max(prob_frames) > 0 else 1.0

    # Scale V to occupy the top 25% of the plot as a subtle background shape.
    # Clip at zero so it never goes negative — avoids visual overlap with the wavefunction.
    V_display = np.clip(V / V_scale * prob_max * 0.25, 0.0, prob_max * 0.25)

    frames = []
    for i, t in enumerate(t_points):
        frames.append(go.Frame(
            data=[
                go.Scatter(x=x, y=V_display, mode="lines",
                           line=dict(color="rgba(120,120,120,0.4)", width=1),
                           fill="tozeroy", fillcolor="rgba(120,120,120,0.07)",
                           showlegend=False),
                go.Scatter(x=x, y=prob_frames[i], mode="lines",
                           line=dict(color="#4C9BE8", width=2.5),
                           fill="tozeroy", fillcolor="rgba(76,155,232,0.2)",
                           showlegend=False),
            ],
            name=str(i),
            layout=go.Layout(annotations=[dict(
                text=f"t = {t:.2f}",
                xref="paper", yref="paper",
                x=0.99, y=0.97,
                xanchor="right", yanchor="top",
                showarrow=False,
                font=dict(size=14, color="rgba(180,180,180,0.9)"),
                bgcolor="rgba(0,0,0,0)",
            )]),
        ))

    # Slider: only label every 10th step to avoid crowding
    slider_steps = []
    for i, t in enumerate(t_points):
        label = f"{t:.0f}" if i % 12 == 0 else " "
        slider_steps.append(dict(
            args=[[str(i)], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
            label=label,
            method="animate",
        ))

    fig = go.Figure(
        data=[
            go.Scatter(x=x, y=V_display, mode="lines",
                       line=dict(color="rgba(120,120,120,0.4)", width=1),
                       fill="tozeroy", fillcolor="rgba(120,120,120,0.07)",
                       name="V(x)"),
            go.Scatter(x=x, y=prob_frames[0], mode="lines",
                       line=dict(color="#4C9BE8", width=2.5),
                       fill="tozeroy", fillcolor="rgba(76,155,232,0.2)",
                       name="|Ψ(x,t)|²"),
        ],
        frames=frames,
    )

    fig.update_layout(
        xaxis_title="Position x",
        yaxis_title="Probability density",
        yaxis_range=[0, prob_max * 1.35],
        height=420,
        margin=dict(l=60, r=30, t=40, b=80),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
        ),
        annotations=[dict(
            text="t = 0.00",
            xref="paper", yref="paper",
            x=0.99, y=0.97,
            xanchor="right", yanchor="top",
            showarrow=False,
            font=dict(size=14, color="rgba(180,180,180,0.9)"),
        )],
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            direction="left",
            xanchor="left",
            yanchor="middle",
            x=0.0,
            y=-0.18,
            pad=dict(r=6, t=0, b=0),
            buttons=[
                dict(label="▶",
                     method="animate",
                     args=[None, dict(frame=dict(duration=80, redraw=True),
                                     fromcurrent=True, mode="immediate")]),
                dict(label="⏸",
                     method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                       mode="immediate")]),
            ],
        )],
        sliders=[dict(
            currentvalue=dict(
                prefix="t = ",
                suffix=" ",
                font=dict(size=13),
                xanchor="center",
            ),
            pad=dict(t=40, b=10),
            x=0.07,
            len=0.93,
            steps=slider_steps,
        )],
    )
    return fig


def build_autocorr_fig(data: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["t_points"], y=data["C_t_abs"],
        mode="lines", line=dict(color="#2ca02c", width=2),
    ))
    fig.update_layout(
        title="|C(t)| — Survival probability",
        xaxis_title="Time",
        yaxis_title="|C(t)|",
        yaxis_range=[0, 1.05],
        height=280,
        margin=dict(l=50, r=20, t=45, b=50),
    )
    return fig


def build_spectrum_fig(data: dict, potential_key: str) -> go.Figure:
    energies = data["energies"]
    amplitudes = data["amplitudes"]

    mask = energies < 20.0
    e_plot = energies[mask]
    a_plot = amplitudes[mask]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=e_plot, y=a_plot,
        mode="lines", line=dict(color="#9467bd", width=1.5),
    ))

    analytic = POTENTIAL_ANALYTIC_ENERGIES.get(potential_key)
    if analytic:
        # Alternate label y position (paper coords) to prevent overlap on closely-spaced levels
        label_y_positions = [1.08, 1.18]
        for i, e_n in enumerate(analytic["levels"]):
            if e_n < 20.0:
                fig.add_shape(
                    type="line",
                    x0=e_n, x1=e_n, y0=0, y1=1,
                    xref="x", yref="paper",
                    line=dict(color="rgba(255,130,50,0.55)", width=1, dash="dot"),
                )
                fig.add_annotation(
                    x=e_n, y=label_y_positions[i % 2],
                    xref="x", yref="paper",
                    text=f"E<sub>{i+1}</sub>",
                    showarrow=False,
                    font=dict(size=11, color="rgba(255,130,50,0.85)"),
                    xanchor="center",
                )

    fig.update_layout(
        title="Excitation spectrum",
        xaxis_title="Energy (natural units)",
        yaxis_title="Amplitude",
        height=280,
        margin=dict(l=50, r=20, t=55, b=50),
    )
    return fig


def build_norm_fig(data: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["t_points"], y=data["norms"],
        mode="lines", line=dict(color="#d62728", width=2),
    ))
    fig.add_hline(y=1.0, line=dict(color="gray", width=1, dash="dash"),
                  annotation_text="Expected", annotation_position="top right")
    fig.update_layout(
        title="∫|Ψ|²dx — Norm conservation",
        xaxis_title="Time",
        yaxis_title="∫|Ψ|²dx",
        yaxis_range=[0, 1.2],
        height=280,
        margin=dict(l=50, r=20, t=45, b=50),
    )
    return fig


def build_expectation_fig(data: dict) -> go.Figure:
    t = data["t_points"]
    x_mean = data["x_mean"]
    p_mean = data["p_mean"]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=t, y=x_mean, mode="lines",
                   line=dict(color="#1f77b4", width=2), name="⟨x⟩(t)"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=t, y=p_mean, mode="lines",
                   line=dict(color="#ff7f0e", width=2), name="⟨p⟩(t)"),
        secondary_y=True,
    )

    fig.update_layout(
        title="Expectation values ⟨x⟩ and ⟨p⟩",
        height=280,
        margin=dict(l=50, r=55, t=45, b=50),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="⟨x⟩ (position)", secondary_y=False,
                     title_font=dict(color="#1f77b4"), tickfont=dict(color="#1f77b4"))
    fig.update_yaxes(title_text="⟨p⟩ (momentum)", secondary_y=True,
                     title_font=dict(color="#ff7f0e"), tickfont=dict(color="#ff7f0e"),
                     showgrid=False)
    return fig


def render_tutorial_sidebar():
    with st.sidebar.expander("What is this?", expanded=False):
        st.markdown("""
**Pinn-Qushion** solves the time-dependent Schrödinger equation (TDSE) using a
Physics-Informed Neural Network (PINN). Select a potential, set the initial wavepacket
parameters, and watch quantum mechanics unfold.
        """)

    with st.sidebar.expander("The Schrödinger equation", expanded=False):
        st.markdown(r"""
The TDSE describes how a quantum state $\Psi(x,t)$ evolves in time:

$$i\hbar \frac{\partial \Psi}{\partial t} = \left[-\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2} + V(x)\right]\Psi$$

Here we use **natural units** where $\hbar = m = 1$. The term $-\frac{1}{2}\partial_{xx}\Psi$
is the kinetic energy; $V(x)\Psi$ is the potential energy.

The initial state is a Gaussian wavepacket:
$$\Psi_0(x) = \left(\frac{1}{2\pi\sigma^2}\right)^{1/4} e^{-(x-x_0)^2/4\sigma^2} e^{ik_0 x}$$

where $x_0$ is the centre and $k_0$ is the initial momentum.
        """)

    with st.sidebar.expander("What is a PINN?", expanded=False):
        st.markdown(r"""
A **Physics-Informed Neural Network** is trained to satisfy a differential equation
as part of its loss function — not just to match data.

The network $\Psi_\theta(x, t; x_0, k_0)$ is penalised for:

- **Physics residual** — how much $i\partial_t\Psi - \hat{H}\Psi$ deviates from zero
- **Initial condition** — how closely it matches the Gaussian at $t=0$
- **Norm conservation** — whether $\int|\Psi|^2 dx = 1$ holds at all times

The key advantage is that the trained network evaluates in microseconds at inference
time, whereas a traditional numerical solver must rerun for each new $(x_0, k_0)$ pair.
        """)

    with st.sidebar.expander("Reading the panels", expanded=False):
        st.markdown(r"""
**Probability density** $|\Psi(x,t)|^2$
The likelihood of finding the particle at position $x$ at time $t$. The shaded
background is a scaled version of $V(x)$ for spatial reference.

**Survival probability** $|C(t)|$
$$C(t) = \int \Psi^*(x,0)\,\Psi(x,t)\,dx$$
Measures how much the current state overlaps with the initial state. A value of 1
means the state has fully returned to its starting configuration. Periodic
recurrences indicate energy quantisation.

**Excitation spectrum** $|{\cal F}[C(t)]|(E)$
The Fourier transform of $C(t)$. Because $C(t) = \sum_n |c_n|^2 e^{-iE_n t}$,
peaks in this spectrum occur at the energy eigenvalues $E_n$ of the potential,
weighted by how strongly the initial wavepacket overlaps with each eigenstate.
Orange lines mark analytic eigenvalues only where the measured amplitude is
significant — levels that are not appreciably excited are not labelled.

**Expectation values** $\langle x \rangle(t)$ and $\langle p \rangle(t)$
$$\langle x \rangle = \int x\,|\Psi|^2\,dx \qquad
\langle p \rangle = \int \Psi^* \!\left(-i\frac{\partial}{\partial x}\right)\!\Psi\,dx$$
Mean position (blue, left axis) and mean momentum (orange, right axis). For a
harmonic oscillator both should oscillate sinusoidally at frequency $\omega$,
90° out of phase — the classical limit of quantum mechanics.

**Norm conservation** $\int|\Psi|^2 dx$
Should equal 1 at all times (the TDSE is unitary). Deviation from 1 reflects
model error — enforcing this is the central challenge in PINN training.
        """)

    with st.sidebar.expander("About the potentials", expanded=False):
        st.markdown(r"""
**Harmonic oscillator** $V(x) = \frac{1}{2}\omega^2 x^2$
Analytically solvable. Eigenvalues $E_n = \omega(n+\frac{1}{2})$. A wavepacket
oscillates without spreading — a "coherent state".

**Infinite square well** $V(x) = 0$ inside, $\infty$ outside
Hard walls force $\Psi = 0$ at boundaries. Eigenvalues $E_n = n^2\pi^2/2L^2$.

**Finite square well**
Like the infinite well but with finite barrier height — the wavefunction
leaks into the classically forbidden region.

**Double well**
Two minima separated by a barrier. A wavepacket initialised in one well
can tunnel through the barrier — a purely quantum mechanical effect.

**Gaussian well** (quantum dot analogue)
A smooth attractive potential. Supports a small number of bound states
depending on depth and width.
        """)


def main():
    st.title("Pinn-Qushion")
    st.markdown(
        "Quantum wavepacket dynamics solved by a Physics-Informed Neural Network. "
        "Select a potential and initial conditions, then press Play."
    )

    manager = get_model_manager()

    # --- Sidebar ---
    st.sidebar.header("Parameters")

    potential_display = st.sidebar.selectbox(
        "Potential",
        options=list(POTENTIAL_DISPLAY_NAMES.keys()),
        index=0,
    )
    potential_key = POTENTIAL_DISPLAY_NAMES[potential_display]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Initial wavepacket")

    x0 = st.sidebar.slider("Initial position x₀", min_value=-5.0, max_value=5.0,
                            value=0.0, step=0.25)
    k0 = st.sidebar.slider("Initial momentum k₀", min_value=-3.0, max_value=3.0,
                            value=1.0, step=0.25)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Potential parameters")
    st.sidebar.markdown(
        f"`{POTENTIAL_PARAMS_DISPLAY[potential_key]}`  \n"
        "<small>Parameters are fixed at trained values. "
        "The network is specific to this potential configuration.</small>",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    render_tutorial_sidebar()

    # --- Check weights ---
    model = manager.get_model(potential_key)
    if model is None:
        st.error(f"No trained weights found for **{potential_display}**. "
                 "Run `python scripts/train_all.py` to train this model.")
        return

    # --- Compute evolution (cached) ---
    with st.spinner("Computing time evolution..."):
        data = compute_evolution(potential_key, x0, k0)

    # --- Animation ---
    st.plotly_chart(build_animation(data, potential_key), use_container_width=True)

    # --- Analysis row ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.plotly_chart(build_autocorr_fig(data), use_container_width=True)
        st.caption(
            "Overlap of the evolving state with its initial configuration. "
            "Periodic recurrences indicate energy quantisation."
        )

    with col2:
        st.plotly_chart(build_spectrum_fig(data, potential_key), use_container_width=True)
        analytic = POTENTIAL_ANALYTIC_ENERGIES.get(potential_key)
        caption = (
            "Fourier transform of the survival probability. "
            "Peaks occur at the energy eigenvalues excited by the initial wavepacket."
        )
        if analytic:
            caption += f" Orange lines: analytic eigenvalues ({analytic['formula']})."
        st.caption(caption)

    with col3:
        st.plotly_chart(build_expectation_fig(data), use_container_width=True)
        st.caption(
            "Mean position (blue) and mean momentum (orange). "
            "For a harmonic oscillator both oscillate sinusoidally at frequency ω, "
            "90° out of phase."
        )

    with col4:
        st.plotly_chart(build_norm_fig(data), use_container_width=True)
        st.caption(
            "Total probability must remain 1 at all times (TDSE is unitary). "
            "Deviation from 1 reflects model accuracy."
        )

    st.markdown("---")
    st.caption(
        "Built with JAX · Equinox · Optax · Streamlit · Plotly  |  "
        "Source: [github.com/Jones-Robert-M/pinn-qushion](https://github.com/Jones-Robert-M/pinn-qushion)"
    )


if __name__ == "__main__":
    main()
