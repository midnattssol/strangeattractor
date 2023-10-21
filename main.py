#!/usr/bin/env python3.11
""""""
import contextlib as ctx
import dataclasses as dc
import functools as ft
import itertools as it
import operator as op

import cairo
import colour
import more_itertools as mit
import numpy as np
from attractors import *


def read_gpl(file):
    lines = file.readlines()
    lines = [line for line in lines if "\t" in line]
    lines = ["#" + line.split()[-1] for line in lines]
    lines = list(map(colour.hex2rgb, lines))

    return np.array(lines)


def sample_unit_disc(n_points, radius=1):
    points = []
    count = 0
    batch_size = 1024

    while len(points) < n_points:
        x_batch = np.random.uniform(-radius, radius, batch_size)
        y_batch = np.random.uniform(-radius, radius, batch_size)

        mask = x_batch**2 + y_batch**2 <= radius**2

        accepted_points = np.column_stack((x_batch[mask], y_batch[mask]))
        points.extend(accepted_points)

    return np.array(points[:n_points])


def sample_unit_circle(num_points):
    # Generate random angles
    angles = np.random.uniform(0, 2 * np.pi, num_points)

    # Compute x and y coordinates on the unit circle
    x_coords = np.cos(angles)
    y_coords = np.sin(angles)

    # Create a numpy array with the points
    points = np.column_stack((x_coords, y_coords))

    return points


def iterate(initial, func, n_iterations=100):
    # Iterate the function some bounded number of times.
    series = mit.iterate(func, initial)
    series = it.islice(series, n_iterations)
    series = zip(*series)
    series = list(series)
    series = np.array(series)

    return series


def is_strange_attractor(func, epsilon=0.0001, upper_bound=5):
    n_samples = 100
    n_iterations = 1000
    initial = sample_unit_disc(n_samples)
    perturbations = sample_unit_circle(n_samples) * epsilon

    series = iterate(initial, func, n_iterations)

    def series_oscillates():
        # Check if the series oscillates by calculating the Lyapunov exponent.
        series_perturbed = iterate(initial + perturbations, func, n_iterations)

        final_diff = (series - series_perturbed)[:, -1, :]
        final_diff = np.linalg.norm(final_diff, axis=1)

        # Since the original distance is always gonna be epsilon, we don't have to calculate it again.
        log_epsilon = np.log(epsilon)

        # Let t be the number of iterations. Then, the Lyapunov exponent λ is defined as the limit as t goes to infinity of the average of the logarithm of the distance between the two points after t steps over the starting distance.
        lyapunov_exponent = np.sum(np.log(final_diff) - log_epsilon) / n_iterations
        return lyapunov_exponent < 0

    def series_converges():
        differences = np.diff(series, axis=1)
        norms = np.linalg.norm(differences, axis=2)
        return np.min(norms) < epsilon

    def series_explodes():
        norms = np.linalg.norm(series, axis=2)
        return np.max(norms) > upper_bound

    return not (series_converges() or series_explodes() or series_oscillates())


def lerp(a, b, t):
    return (b - a) * t + a


def distance_to_closest_neighbor_high_memory_usage(a):
    dist = (a[:, None, :] - a[None, :, :]).astype("float")
    dist = np.sum(dist**2, axis=-1)
    np.fill_diagonal(dist, "inf")
    closest_dists = np.min(dist, axis=1) ** 0.5
    return closest_dists


def distance_to_closest_neighbor(a):
    out = []
    chunk_size = 300
    i = 0

    while i * chunk_size < len(a):
        dist = a[:, None, :] - a[i * chunk_size : (i + 1) * chunk_size, :]
        dist = np.sum(dist**2, axis=-1)

        # incorrect  - dodges 0s but also often accidentally drops other points
        o = np.partition(dist, 1, axis=1)[:, 1]
        out.extend(o)
        i += 1

    return np.array(out)


@dc.dataclass
class AttractorPlotter:
    attractor: callable
    data: np.array = None
    context: cairo.Context = None

    n_samples: int = 415_000
    n_iterations: int = 12
    n_skip_iterations: int = 1

    background_color: list = (0, 0, 0.02)

    scale: float = 1
    size: int = 4000
    style: str = "dynamic-width"
    palette: list = None

    _palette_lookup_table: None = None

    def make_lookup_table(self, n=100):
        items = self.palette
        out = []

        for i in range(n):
            frac = (i / n) * len(items)
            idx = int(frac)

            if (idx + 1) >= len(items):
                break

            out.append(lerp(items[idx], items[idx + 1], frac - idx))

        self._palette_lookup_table = out[::-1]

    @property
    def n_points(self):
        return self.n_samples * (self.n_iterations - self.n_skip_iterations)

    def render_to(self, path):
        self.make_lookup_table()
        print(f"Rendering attractor to {path} with {self.n_points} points...")

        # Randomly sample points, iterate them, and ignore the first few (not rigorous)
        # to only gain items within the attractor. This is done to exploit the vectorization
        # of the functions - the iteration obviously requires the previous value to be
        # known, so this allows some form of pararellism.
        samples = sample_unit_disc(self.n_samples).astype(np.float16)
        points = iterate(samples, self.attractor, self.n_iterations)
        points = points[:, self.n_skip_iterations :, :]

        # points = np.reshape(points, (-1, 2))
        # points = [points]

        with self.rendering_to(path):
            for seq in points:
                self.drawseq(seq)

    @ctx.contextmanager
    def rendering_to(self, output_file):
        # Create a new surface and context
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.size, self.size)
        context = cairo.Context(surface)

        # Scale the coordinate system.
        context.scale(self.size / 2, self.size / 2)
        context.set_line_width(0.01)

        # Translate the coordinate system so (0, 0) is at the center.
        context.translate(1, 1)

        # Set background color.
        context.set_source_rgb(*self.background_color)
        context.rectangle(-1, -1, 2, 2)
        context.fill()

        # Scale down a little to make the unit circle not take up all the space.
        context.scale(0.4, 0.4)
        context.scale(self.scale, self.scale)

        self.context = context
        yield

        # Save to PNG
        surface.write_to_png(output_file)
        print(f"Image saved to {output_file}.")
        self.context = None

    def drawseq(self, points):
        length_diffs = np.diff(points, axis=0)
        length_diffs = np.linalg.norm(length_diffs, axis=1)

        maxlen = np.max(length_diffs)
        maxlen = maxlen if maxlen != 0 else 1

        length_diffs /= maxlen

        # The arc width scales with the square of the number of points,
        # since the square of the width is proportional to the area covered.
        point_radius_scale = 2
        point_radius = point_radius_scale / np.sqrt(self.n_points)
        alpha = 0.3

        alpha_scaling = 0.12

        resolution = len(self._palette_lookup_table) - 1

        if self.style == "dynamic-width":
            # Note that these distances are only an approximation
            # based on the current batch. The distances may vary a lot if multiple
            # batches are used.
            distances = distance_to_closest_neighbor(points)
            maxlen = np.max(distances)
            maxlen = maxlen if maxlen != 0 else 1

            distances /= maxlen

            alphas = alpha / distances
            alphas = np.clip(alphas, 0, 1)

        # Draw the line with color gradient
        for i in range(len(points) - 1):
            color = self._palette_lookup_table[int(resolution * length_diffs[i])]

            if self.style == "lines":
                self.context.set_source_rgba(*color, 0.1)
                self.context.move_to(*points[i])
                self.context.line_to(*points[i + 1])
                self.context.stroke()

            elif self.style == "fixed-width":
                self.context.set_source_rgba(*color, 1)
                self.context.arc(*points[i], 0.02, 0, 2 * np.pi)

            elif self.style == "dynamic-width":
                if distances[i] == 0:
                    continue

                # color = self._palette_lookup_table[int(distances[i] * len(self._palette_lookup_table) * 0.99)]
                opacity = alphas[i] * alpha_scaling
                self.context.set_source_rgba(*color, opacity)

                # Render full:
                # radius = (point_radius * (0.2**-0.8)) * 4

                radius = (point_radius * (alphas[i] ** -0.8)) * 2
                self.context.arc(*points[i], radius, 0, 2 * np.pi)
                self.context.fill()

            else:
                raise NotImplementedError()
