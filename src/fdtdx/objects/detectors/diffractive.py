import jax
import jax.numpy as jnp
import pytreeclass as tc
from typing import Sequence, Literal, Tuple

from fdtdx.core.physics import constants
from fdtdx.objects.detectors.detector import Detector, DetectorState

@tc.autoinit
class DiffractiveDetector(Detector):
    """Detector for computing Fourier transforms of fields at specific frequencies and diffraction orders.
    
    This detector is similar to Tidy3D's DiffractionMonitor, computing field amplitudes for specific
    diffraction orders and frequencies. It performs spatial and temporal Fourier transforms to analyze
    the diffracted field coefficients in a specified plane.

    Attributes:
        frequencies: List of frequencies to analyze (in Hz)
        orders: Tuple of (nx, ny) pairs specifying diffraction orders to compute
        plane: Plane for diffraction analysis ('xy', 'yz', or 'xz')
        direction: Direction of diffraction analysis ("+" or "-")
        as_slices: If True, returns results as 2D slices rather than full volume
        reduce_volume: If True, reduces volume data to single values
    """
    
    frequencies: Sequence[float] = (0.)
    orders: Sequence[Tuple[int, int]] = ((0, 0))
    plane: Literal['xy', 'yz', 'xz'] = "xy"
    direction: Literal["+", "-"] = "+"
    as_slices: bool = False
    reduce_volume: bool = False
    dtype: jnp.dtype = tc.field(
        default=jnp.complex64,
        kind="KW_ONLY",
    )

    def __post_init__(self):
        if self.dtype not in [jnp.complex64, jnp.complex128]:
            raise Exception(f"Invalid dtype in DiffractiveDetector: {self.dtype}")
        
        # Set normal axis based on plane
        self._normal_axis = {'xy': 2, 'yz': 0, 'xz': 1}[self.plane]
        
        # Precompute angular frequencies for vectorization
        self._angular_frequencies = 2 * jnp.pi * jnp.array(self.frequencies)
        
        # Initialize order info attributes as None
        self._kx_indices = None
        self._ky_indices = None
        self._Nx = None
        self._Ny = None
        self._kx_normalized = None
        self._ky_normalized = None
        
    def _precompute_order_info(self):
        """Precompute order indices and wavevectors for faster processing."""
        # Get grid dimensions for the plane
        if self.plane == 'xy':
            Nx, Ny = self.grid_shape[0], self.grid_shape[1]
        elif self.plane == 'yz':
            Nx, Ny = self.grid_shape[1], self.grid_shape[2]
        else:  # xz plane
            Nx, Ny = self.grid_shape[0], self.grid_shape[2]
            
        # Convert orders to array for vectorization
        orders = jnp.array(self.orders)  # Shape: (num_orders, 2)
        
        # Compute FFT indices for all orders at once
        kx_indices = jnp.where(orders[:, 0] >= 0, 
                             orders[:, 0], 
                             Nx + orders[:, 0])
        ky_indices = jnp.where(orders[:, 1] >= 0, 
                             orders[:, 1], 
                             Ny + orders[:, 1])
        
        # Store precomputed values
        self._kx_indices = kx_indices
        self._ky_indices = ky_indices
        self._Nx = Nx
        self._Ny = Ny
        
        # Precompute normalized wavevectors for physical validation
        dx = dy = self._config.resolution
        self._kx_normalized = 2 * jnp.pi * orders[:, 0] / (Nx * dx)
        self._ky_normalized = 2 * jnp.pi * orders[:, 1] / (Ny * dy)
        
    @property
    def propagation_axis(self) -> int:
        """Determines the axis normal to the detector plane.

        Returns:
            int: Index of the normal axis based on the specified plane
        """
        return self._normal_axis
    
    def _validate_orders(self, wavelength: float) -> None:
        """Validate that requested diffraction orders are physically realizable.
        
        Args:
            wavelength: Wavelength of the light in meters
            
        Raises:
            Exception: If any requested order is not physically realizable
        """
        if self._Nx is None:
            raise Exception("Order info not yet computed. Run update first.")
            
        # Maximum possible orders based on grid
        max_nx = self._Nx // 2
        max_ny = self._Ny // 2
        
        # Check Nyquist limits for all orders at once
        nx_valid = jnp.all(jnp.abs(jnp.array([o[0] for o in self.orders])) <= max_nx)
        ny_valid = jnp.all(jnp.abs(jnp.array([o[1] for o in self.orders])) <= max_ny)
        
        if not (nx_valid and ny_valid):
            raise Exception(
                f"Some orders exceed Nyquist limit for grid size ({self._Nx}, {self._Ny})"
            )
        
        # Check physical realizability for all orders at once
        k0 = 2 * jnp.pi / wavelength
        kt_squared = self._kx_normalized**2 + self._ky_normalized**2
        
        if jnp.any(kt_squared > k0**2):
            raise Exception(
                f"Some orders are evanescent at wavelength {wavelength*1e9:.1f}nm"
            )

    def _shape_dtype_single_time_step(self) -> dict[str, jax.ShapeDtypeStruct]:
        """Define shape and dtype for a single time step of diffractive data.

        Returns:
            dict: Dictionary mapping data keys to ShapeDtypeStruct containing shape and
                dtype information for each frequency and order combination.
        """
        num_freqs = len(self.frequencies)
        num_orders = len(self.orders)
        
        if self.reduce_volume:
            shape = (num_freqs, num_orders)
        else:
            shape = (num_freqs, num_orders, *self.grid_shape)
            
        # Ensure we're using a complex dtype
        field_dtype = jnp.complex128 if self.dtype == jnp.float64 else jnp.complex64
        return {"diffractive": jax.ShapeDtypeStruct(shape=shape, dtype=field_dtype)}

    def _num_latent_time_steps(self) -> int:
        """Get number of time steps needed for latent computation.

        Returns:
            int: Always returns 1 for diffractive detector since only current state is needed.
        """
        return 1

    def update(
        self,
        time_step: jax.Array,
        E: jax.Array,
        H: jax.Array,
        state: DetectorState,
        inv_permittivity: jax.Array,
        inv_permeability: jax.Array,
    ) -> DetectorState:
        """Update the diffractive detector state with current field values.

        Computes spatial and temporal Fourier transforms of the fields to analyze
        diffraction orders at specified frequencies.

        Args:
            time_step: Current simulation time step
            E: Electric field array
            H: Magnetic field array
            state: Current detector state
            inv_permittivity: Inverse permittivity array
            inv_permeability: Inverse permeability array

        Returns:
            DetectorState: Updated state containing new diffractive values
        """
        del inv_permittivity, inv_permeability
        
        # Precompute order info if not done yet
        if self._kx_indices is None:
            self._precompute_order_info()
        
        # Get current field values at the specified plane using base class's grid_slice
        cur_E = E[:, *self.grid_slice]  # Shape: (3, nx, ny, 1)
        cur_H = H[:, *self.grid_slice]  # Shape: (3, nx, ny, 1)
        
        # Remove the normal axis dimension since it should be 1
        cur_E = jnp.squeeze(cur_E, axis=self.propagation_axis + 1)  # Shape: (3, nx, ny)
        cur_H = jnp.squeeze(cur_H, axis=self.propagation_axis + 1)  # Shape: (3, nx, ny)
        
        # Compute total field intensity (|E|^2 + |H|^2) - vectorized
        field_intensity = jnp.sum(jnp.abs(cur_E)**2 + jnp.abs(cur_H)**2, axis=0)
        
        # Compute spatial FFT of the total field
        spatial_fft = jnp.fft.fft2(field_intensity)
        
        # Extract all orders at once using advanced indexing
        order_amplitudes = spatial_fft[self._kx_indices, self._ky_indices]  # Shape: (num_orders,)
        
        # Time domain analysis - vectorized for all frequencies
        t = time_step * self._config.time_step_duration
        phase_angles = self._angular_frequencies[:, None] * t  # Shape: (num_freqs, 1)
        phasors = jnp.exp(-1j * phase_angles)  # Shape: (num_freqs, 1)
        
        # Compute all frequency components for all orders at once
        order_amplitudes = order_amplitudes[None, :]  # Shape: (1, num_orders)
        new_values = order_amplitudes * phasors  # Shape: (num_freqs, num_orders)
        
        # Update state
        arr_idx = self._time_step_to_arr_idx[time_step]
        new_state = state.copy()
        
        if not self.reduce_volume:
            new_values = new_values.reshape(len(self.frequencies), len(self.orders), *self.grid_shape)
            
        new_state["diffractive"] = new_state["diffractive"].at[arr_idx].set(new_values)
        
        return new_state 