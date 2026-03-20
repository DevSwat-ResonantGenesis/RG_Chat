"""
Resonance Hashing Service
Hashes text by meaning, energy, and spin to create resonance signatures
Includes XYZ coordinate system for 3D semantic space
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA


class ResonanceHasher:
    """Hash text by meaning, energy, and spin"""
    
    # Cache for PCA model (reused across calls)
    _pca_model: Optional[PCA] = None
    
    # Resonance function constants (from foundational architecture)
    RESONANCE_A = np.pi / 4  # sin coefficient
    RESONANCE_B = np.e / 3    # cos coefficient
    RESONANCE_C = 1.618 / 2   # tan coefficient (golden ratio / 2)
    
    @staticmethod
    def hash_to_universe_id(text: str) -> str:
        """
        Convert text to Universe ID (256-bit hash).
        
        Mathematical definition: u = H(x)
        Where H is SHA-256 cryptographic hash.
        
        Args:
            text: Input text
        
        Returns:
            Universe ID as hex string (64 characters)
        """
        return hashlib.sha256(text.encode()).hexdigest()
    
    @staticmethod
    def universe_id_to_vector(universe_id: str) -> np.ndarray:
        """
        Convert Universe ID to normalized vector.
        
        Mathematical definition: v_h = int(u) / ||int(u)||
        
        Args:
            universe_id: 256-bit hash as hex string
        
        Returns:
            Normalized vector (numpy array)
        """
        # Convert hex to integer
        int_val = int(universe_id, 16)
        
        # Convert to bytes (32 bytes for 256 bits)
        bytes_val = int_val.to_bytes(32, 'big')
        
        # Convert to numpy array (uint8)
        vec = np.frombuffer(bytes_val, dtype=np.uint8).astype(np.float64)
        
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        return vec
    
    @staticmethod
    def hash_text(text: str, context: Optional[str] = None) -> str:
        """
        Create a resonance hash from text.
        
        The hash encodes:
        - Meaning: Semantic content
        - Energy: Emotional/intensity level
        - Spin: Direction/intent
        """
        # Normalize text
        normalized = text.lower().strip()
        
        # Extract meaning (semantic hash)
        meaning_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        
        # Extract energy (emotional/intensity level)
        energy_score = ResonanceHasher._calculate_energy(normalized)
        energy_hash = hashlib.sha256(str(energy_score).encode()).hexdigest()[:8]
        
        # Extract spin (direction/intent)
        spin_score = ResonanceHasher._calculate_spin(normalized)
        spin_hash = hashlib.sha256(str(spin_score).encode()).hexdigest()[:8]
        
        # Combine with context if provided
        if context:
            context_hash = hashlib.sha256(context.lower().encode()).hexdigest()[:8]
            combined = f"{meaning_hash}-{energy_hash}-{spin_hash}-{context_hash}"
        else:
            combined = f"{meaning_hash}-{energy_hash}-{spin_hash}"
        
        # Final hash
        final_hash = hashlib.sha256(combined.encode()).hexdigest()
        return final_hash
    
    @staticmethod
    def _calculate_energy(text: str) -> float:
        """Calculate energy level (0-1) based on emotional/intensity indicators"""
        # Simple heuristic: count intensity words, punctuation, caps
        intensity_words = ['very', 'extremely', 'highly', 'critical', 'urgent', 'important', 'essential']
        intensity_count = sum(1 for word in intensity_words if word in text.lower())
        
        # Punctuation intensity
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        # Caps intensity
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        # Normalize to 0-1
        energy = min(1.0, (intensity_count * 0.1 + exclamation_count * 0.05 + 
                          question_count * 0.03 + caps_ratio * 0.5))
        return round(energy, 3)
    
    @staticmethod
    def _calculate_spin(text: str) -> float:
        """Calculate spin (direction/intent) - positive/negative/neutral"""
        # Simple sentiment-based spin
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'happy', 'success']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'sad', 'fail', 'error', 'problem']
        
        positive_count = sum(1 for word in positive_words if word in text.lower())
        negative_count = sum(1 for word in negative_words if word in text.lower())
        
        # Spin: -1 (negative) to +1 (positive), normalized to 0-1
        if positive_count + negative_count == 0:
            spin = 0.5  # Neutral
        else:
            spin_raw = (positive_count - negative_count) / (positive_count + negative_count)
            spin = (spin_raw + 1) / 2  # Normalize to 0-1
        
        return round(spin, 3)
    
    @staticmethod
    def calculate_resonance(hash1: str, hash2: str) -> float:
        """Calculate resonance score between two hashes (0-1)"""
        # Simple hamming distance on hash strings
        if len(hash1) != len(hash2):
            return 0.0
        
        matches = sum(1 for a, b in zip(hash1, hash2) if a == b)
        similarity = matches / len(hash1)
        return round(similarity, 3)
    
    @staticmethod
    def extract_anchors(text: str, max_anchors: int = 5) -> List[str]:
        """Extract key phrases/anchors from text"""
        # Simple extraction: key phrases, important words
        # In production, use NLP models for better extraction
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                     'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        words = text.lower().split()
        important_words = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Extract phrases (2-3 word combinations)
        anchors = []
        for i in range(len(important_words) - 1):
            phrase = f"{important_words[i]} {important_words[i+1]}"
            if len(phrase) > 5:  # Minimum phrase length
                anchors.append(phrase)
        
        # Also add single important words
        anchors.extend([w for w in important_words if len(w) > 5][:max_anchors])
        
        # Remove duplicates and limit
        unique_anchors = list(dict.fromkeys(anchors))[:max_anchors]
        return unique_anchors
    
    @staticmethod
    def calculate_xyz_coordinates(embedding: List[float]) -> Tuple[float, float, float]:
        """
        Convert embedding vector to 3D XYZ coordinates in semantic space.
        
        Uses PCA (Principal Component Analysis) to reduce embedding dimensions to 3D.
        This creates a 3D semantic space where similar meanings are close together.
        
        Args:
            embedding: High-dimensional embedding vector (e.g., 384, 768, 1536 dimensions)
        
        Returns:
            Tuple of (x, y, z) coordinates in 3D semantic space
        """
        if not embedding or len(embedding) < 3:
            # Fallback: use hash-based coordinates if no embedding
            hash_val = hashlib.sha256(str(embedding).encode()).hexdigest()
            x = int(hash_val[:8], 16) / 0xFFFFFFFF  # Normalize to 0-1
            y = int(hash_val[8:16], 16) / 0xFFFFFFFF
            z = int(hash_val[16:24], 16) / 0xFFFFFFFF
            return (x, y, z)
        
        # Convert to numpy array
        embedding_array = np.array(embedding).reshape(1, -1)
        
        # If embedding is already 3D or less, use directly (normalized)
        if len(embedding) <= 3:
            coords = embedding_array[0]
            # Normalize to 0-1 range
            coords = (coords - coords.min()) / (coords.max() - coords.min() + 1e-10)
            # Pad or truncate to 3D
            if len(coords) < 3:
                coords = np.pad(coords, (0, 3 - len(coords)), mode='constant')
            return tuple(coords[:3].tolist())
        
        # Use PCA to reduce to 3D
        # Initialize PCA model if not exists
        if ResonanceHasher._pca_model is None:
            ResonanceHasher._pca_model = PCA(n_components=3)
            # Fit on a dummy embedding to initialize (will be refit on actual data)
            dummy = np.random.rand(1, len(embedding))
            ResonanceHasher._pca_model.fit(dummy)
        
        # Transform embedding to 3D
        coords_3d = ResonanceHasher._pca_model.transform(embedding_array)[0]
        
        # Normalize to 0-1 range for consistent semantic space
        # Use min-max normalization
        coords_3d = (coords_3d - coords_3d.min()) / (coords_3d.max() - coords_3d.min() + 1e-10)
        
        return tuple(coords_3d.tolist())
    
    @staticmethod
    def calculate_proximity(xyz1: Tuple[float, float, float], xyz2: Tuple[float, float, float]) -> float:
        """
        Calculate Euclidean distance between two points in 3D semantic space.
        
        Distance formula: √[(x1-x2)² + (y1-y2)² + (z1-z2)²]
        
        Args:
            xyz1: First point (x, y, z)
            xyz2: Second point (x, y, z)
        
        Returns:
            Distance (0 = same point, higher = more distant)
        """
        x1, y1, z1 = xyz1
        x2, y2, z2 = xyz2
        
        distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        return float(distance)
    
    @staticmethod
    def calculate_proximity_score(xyz1: Tuple[float, float, float], xyz2: Tuple[float, float, float]) -> float:
        """
        Calculate proximity score (0-1) where 1 = same point, 0 = very distant.
        
        Converts distance to similarity score.
        
        Args:
            xyz1: First point (x, y, z)
            xyz2: Second point (x, y, z)
        
        Returns:
            Proximity score (0-1, higher = more similar)
        """
        distance = ResonanceHasher.calculate_proximity(xyz1, xyz2)
        # Convert distance to similarity score (inverse relationship)
        # Using exponential decay: score = e^(-distance)
        # This gives 1.0 for distance 0, ~0.37 for distance 1, ~0.14 for distance 2
        score = np.exp(-distance)
        return float(score)
    
    @staticmethod
    def apply_spin(
        point: np.ndarray,
        spin_vector: Optional[np.ndarray] = None,
        spin_angle: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply spin (internal semantic rotation) to a point.
        
        Mathematical definition: s_{t+1} = R(ω) · s_t
        
        Where:
        - R(ω) = rotation matrix
        - ω = spin vector from semantic volatility
        - s_t = current point position
        
        Args:
            point: Current point in 3D space (x, y, z)
            spin_vector: Optional spin vector (if None, uses random small rotation)
            spin_angle: Optional rotation angle in radians (if None, uses spin_vector magnitude)
        
        Returns:
            Rotated point
        """
        if isinstance(point, (list, tuple)):
            point = np.array(point)
        
        # If no spin vector provided, use small random rotation
        if spin_vector is None:
            # Small random rotation (semantic drift)
            spin_vector = np.random.normal(0, 0.01, size=3)
        
        if isinstance(spin_vector, (list, tuple)):
            spin_vector = np.array(spin_vector)
        
        # Calculate rotation angle
        if spin_angle is None:
            angle = np.linalg.norm(spin_vector)
            if angle < 1e-10:
                return point  # No rotation
            axis = spin_vector / angle
        else:
            angle = spin_angle
            if np.linalg.norm(spin_vector) > 1e-10:
                axis = spin_vector / np.linalg.norm(spin_vector)
            else:
                axis = np.array([0, 0, 1])  # Default axis
        
        # Rodrigues' rotation formula for 3D rotation
        # R = I + sin(θ)K + (1 - cos(θ))K²
        # where K is the cross-product matrix of axis
        
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        # Cross-product matrix K
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        # Rotation matrix
        I = np.eye(3)
        R = I + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
        
        # Apply rotation
        rotated_point = np.dot(R, point)
        
        # Normalize to keep on unit sphere
        norm = np.linalg.norm(rotated_point)
        if norm > 0:
            rotated_point = rotated_point / norm
        
        return rotated_point
    
    @staticmethod
    def apply_drift(
        point: np.ndarray,
        anchor: np.ndarray,
        gamma: float = 0.1
    ) -> np.ndarray:
        """
        Apply drift (decay toward anchor).
        
        Mathematical definition: s_{t+1} = s_t + γ(A_{j*} - s_t)
        
        Where:
        - s_t = current point position
        - A_{j*} = anchor position
        - γ ∈ [0,1] = drift coefficient (decay rate)
        
        Args:
            point: Current point in 3D space (x, y, z)
            anchor: Anchor point to drift toward
            gamma: Drift coefficient (0-1, higher = faster drift)
        
        Returns:
            Drifted point (closer to anchor)
        """
        if isinstance(point, (list, tuple)):
            point = np.array(point)
        if isinstance(anchor, (list, tuple)):
            anchor = np.array(anchor)
        
        # Clamp gamma to [0, 1]
        gamma = max(0.0, min(1.0, gamma))
        
        # Apply drift: s_{t+1} = s_t + γ(A - s_t)
        drifted_point = point + gamma * (anchor - point)
        
        # Normalize to keep on unit sphere
        norm = np.linalg.norm(drifted_point)
        if norm > 0:
            drifted_point = drifted_point / norm
        
        return drifted_point
    
    @staticmethod
    def calculate_stability(
        point: np.ndarray,
        previous_points: List[np.ndarray],
        threshold: float = 0.1
    ) -> float:
        """
        Calculate stability metric for a point.
        
        Stability measures how much a point has moved over time.
        Lower values = more stable (less movement)
        Higher values = less stable (more movement)
        
        Args:
            point: Current point position
            previous_points: List of previous positions
            threshold: Distance threshold for stability
        
        Returns:
            Stability score (0-1, lower = more stable)
        """
        if not previous_points:
            return 0.0  # No history = stable
        
        if isinstance(point, (list, tuple)):
            point = np.array(point)
        
        # Calculate average distance from previous points
        distances = []
        for prev_point in previous_points:
            if isinstance(prev_point, (list, tuple)):
                prev_point = np.array(prev_point)
            distance = np.linalg.norm(point - prev_point)
            distances.append(distance)
        
        avg_distance = np.mean(distances)
        
        # Normalize to 0-1 (stability score)
        # Higher distance = less stable
        stability_score = min(1.0, avg_distance / threshold)
        
        return float(stability_score)
    
    @staticmethod
    def to_hyperspherical(coords: np.ndarray) -> Dict[str, float]:
        """
        Convert 3D coordinates to hyperspherical coordinates (latitude/longitude).
        
        Mathematical definition:
        For s = (x, y, z):
        φ = arcsin(z)  # Latitude
        θ = arctan2(y, x)  # Longitude
        r = ||s||  # Radius (should be 1 for unit sphere)
        
        Args:
            coords: 3D coordinates (x, y, z)
        
        Returns:
            Dictionary with 'r' (radius), 'phi' (latitude), 'theta' (longitude)
        """
        if isinstance(coords, (list, tuple)):
            coords = np.array(coords)
        
        x, y, z = coords[0], coords[1], coords[2]
        
        # Radius
        r = np.linalg.norm(coords)
        
        # Latitude (phi) - angle from equator
        phi = np.arcsin(np.clip(z / r if r > 0 else 0, -1, 1))
        
        # Longitude (theta) - angle around equator
        theta = np.arctan2(y, x)
        
        return {
            'r': float(r),
            'phi': float(phi),  # Latitude in radians
            'theta': float(theta),  # Longitude in radians
        }
    
    @staticmethod
    def from_hyperspherical(r: float, phi: float, theta: float) -> np.ndarray:
        """
        Convert hyperspherical coordinates back to 3D Cartesian coordinates.
        
        Mathematical definition:
        x = r * cos(φ) * cos(θ)
        y = r * cos(φ) * sin(θ)
        z = r * sin(φ)
        
        Args:
            r: Radius
            phi: Latitude (in radians)
            theta: Longitude (in radians)
        
        Returns:
            3D coordinates (x, y, z)
        """
        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.cos(phi) * np.sin(theta)
        z = r * np.sin(phi)
        
        return np.array([x, y, z])
    
    @staticmethod
    def fuse_hash_and_embedding(
        hash_vector: np.ndarray,
        embedding: np.ndarray,
        alpha: float = 0.9
    ) -> np.ndarray:
        """
        Fuse hash identity with semantic meaning.
        
        Mathematical definition: s₀ = α·ê + (1-α)·v_h
        Then: s = s₀ / ||s₀|| (project onto unit sphere)
        
        Where:
        - α ∈ [0,1] controls meaning vs identity (usually 0.85-0.95)
        - ê = normalized embedding (semantic meaning)
        - v_h = normalized hash vector (deterministic identity)
        
        Args:
            hash_vector: Normalized hash vector v_h
            embedding: Embedding vector (will be normalized)
            alpha: Fusion weight (default 0.9 = 90% meaning, 10% identity)
        
        Returns:
            Fused sphere point s (normalized to unit sphere)
        """
        # Normalize embedding
        embed_norm = np.array(embedding, dtype=np.float64)
        embed_norm = embed_norm / np.linalg.norm(embed_norm)
        
        # Ensure hash_vector is normalized
        hash_norm = hash_vector / np.linalg.norm(hash_vector) if np.linalg.norm(hash_vector) > 0 else hash_vector
        
        # Fusion: s₀ = α·ê + (1-α)·v_h
        s0 = alpha * embed_norm + (1 - alpha) * hash_norm
        
        # Project onto unit sphere: s = s₀ / ||s₀||
        s = s0 / np.linalg.norm(s0)
        
        return s
    
    @staticmethod
    def calculate_resonance_function(xyz: Tuple[float, float, float]) -> float:
        """
        Calculate resonance using the foundational architecture formula.
        
        Mathematical definition: R(h) = sin(a·x) + cos(b·y) + tan(c·z)
        
        Where:
        - a = π/4 (resonance coefficient for x)
        - b = e/3 (resonance coefficient for y)
        - c = φ/2 (resonance coefficient for z, where φ = golden ratio)
        
        Args:
            xyz: Tuple of (x, y, z) coordinates
        
        Returns:
            Resonance value (can be negative or positive)
        """
        x, y, z = xyz
        
        # Resonance function: R(h) = sin(a·x) + cos(b·y) + tan(c·z)
        resonance = (
            np.sin(ResonanceHasher.RESONANCE_A * x) +
            np.cos(ResonanceHasher.RESONANCE_B * y) +
            np.tan(ResonanceHasher.RESONANCE_C * z)
        )
        
        return float(resonance)
    
    @staticmethod
    def calculate_anchor_energy(
        point: np.ndarray,
        anchor: np.ndarray,
        beta: float = 1.0
    ) -> float:
        """
        Calculate anchor attraction energy.
        
        Mathematical definition: E_j(s) = exp(-β·||s - A_j||²)
        
        Where:
        - β controls "sharpness" of resonance
        - Higher E_j means stronger alignment with anchor A_j
        
        Args:
            point: Sphere point s (normalized vector)
            anchor: Anchor position A_j (normalized vector)
            beta: Sharpness parameter (default 1.0)
        
        Returns:
            Energy value (0-1, higher = stronger alignment)
        """
        # Ensure both are normalized
        point_norm = point / np.linalg.norm(point) if np.linalg.norm(point) > 0 else point
        anchor_norm = anchor / np.linalg.norm(anchor) if np.linalg.norm(anchor) > 0 else anchor
        
        # Calculate squared distance: ||s - A_j||²
        distance_sq = np.sum((point_norm - anchor_norm) ** 2)
        
        # Energy: E_j(s) = exp(-β·||s - A_j||²)
        energy = np.exp(-beta * distance_sq)
        
        return float(energy)
    
    @staticmethod
    def find_best_anchor(
        point: np.ndarray,
        anchors: List[np.ndarray],
        beta: float = 1.0
    ) -> Tuple[int, float]:
        """
        Find anchor with highest energy.
        
        Mathematical definition: j* = argmax_j E_j(s)
        
        Args:
            point: Sphere point s
            anchors: List of anchor positions A_j
            beta: Sharpness parameter
        
        Returns:
            Tuple of (best_anchor_index, best_energy_score)
        """
        if not anchors:
            return (-1, 0.0)
        
        energies = [
            ResonanceHasher.calculate_anchor_energy(point, anchor, beta)
            for anchor in anchors
        ]
        
        best_idx = int(np.argmax(energies))
        best_energy = energies[best_idx]
        
        return (best_idx, best_energy)
    
    @staticmethod
    def hash_to_coords(hash_str: str) -> Tuple[float, float, float]:
        """
        Convert a hash string to 3D XYZ coordinates.
        
        This is a deterministic mapping from hash to coordinates.
        Used for backward compatibility with existing code.
        
        Args:
            hash_str: Hash string (hex)
        
        Returns:
            Tuple of (x, y, z) coordinates
        """
        # Use first 24 characters of hash for coordinates
        # Each coordinate gets 8 hex characters (32 bits)
        if len(hash_str) < 24:
            hash_str = hash_str.ljust(24, '0')
        
        x = int(hash_str[:8], 16) / 0xFFFFFFFF  # Normalize to 0-1
        y = int(hash_str[8:16], 16) / 0xFFFFFFFF
        z = int(hash_str[16:24], 16) / 0xFFFFFFFF
        
        return (x, y, z)

