#!/usr/bin/env python3.11
""""""
import contextlib as ctx
import dataclasses as dc
import functools as ft
import itertools as it
import operator as op

import cairo
import more_itertools as mit
import numpy as np
from attractors import clifford_attractor


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


# @fscache.fscache(".cache", (lambda initial, func, n_iterations: [initial, func.__name__, n_iterations]))
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


@dc.dataclass
class AttractorPlotter:
    attractor: callable
    data: np.array = None
    context: cairo.Context = None

    n_samples: int = 10_000
    n_iterations: int = 100
    n_skip_iterations: int = 5

    @property
    def n_points(self):
        return self.n_samples * (self.n_iterations - self.n_skip_iterations)

    def render_to(self, path):
        # Randomly sample a ton of points and then use them?
        samples = sample_unit_disc(self.n_samples)
        points = iterate(samples, self.attractor, self.n_iterations)

        print(f"Rendering attractor to {path} with {self.n_points} points...")

        with self.rendering_to(path):
            for seq in points:
                self.drawseq(seq[self.n_skip_iterations :])

    @ctx.contextmanager
    def rendering_to(self, output_file):
        # Create a new surface and context
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        context = cairo.Context(surface)

        # Scale the coordinate system
        context.scale(width / 2, height / 2)
        context.set_line_width(0.01)

        # Translate the coordinate system so (0, 0) is at the center
        context.translate(1, 1)

        # Set background color
        context.set_source_rgb(*background_color)
        context.rectangle(-1, -1, 2, 2)
        context.fill()

        # Scale down a little to make the unit circle not take up all the space
        context.scale(0.3, 0.3)

        context.set_source_rgba(*circle_color)
        context.arc(0, 0, 1.5, 0, 2 * np.pi)
        context.fill()

        self.context = context
        yield

        # Save to PNG
        surface.write_to_png(output_file)
        print(f"Image saved to {output_file}")
        self.context = None

    def drawseq(self, points):
        c0 = np.array([0.9, 0.9, 0.85])
        c1 = np.array([0.9, 0.13, 0.1])

        length_diffs = np.diff(points, axis=0)
        length_diffs = np.linalg.norm(length_diffs, axis=1)

        maxlen = np.max(length_diffs)
        maxlen = maxlen if maxlen != 0 else 1

        length_diffs /= maxlen

        # The arc width scales with the square of the number of points,
        # since the square of the width is proportional to the area covered.
        arc_width_scale = 3
        arc_width = arc_width_scale / np.sqrt(self.n_points)
        alpha = 0.3

        # Draw the line with color gradient
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]

            # optimize: use a LUT instead
            color = lerp(c0, c1, length_diffs[i])

            # # Uncomment to use lines:
            # context.set_source_rgba(*color, 0.1)
            # context.move_to(*points[i])
            # context.line_to(*points[i + 1])
            # context.line_to(*lerp(points[i], points[i + 1], 0.1))
            # context.stroke()

            self.context.set_source_rgba(*color, alpha)
            self.context.arc(*points[i], arc_width, 0, 2 * np.pi)
            self.context.fill()


# ===| Previously test module |===


# Image parameters
width, height = 4000, 4000
output_file = "output.png"

# Colors
background_color = (0, 0, 0)  # Black background
circle_color = (0, 0.02, 0.03)  # Semi-transparent black


def main():
    np.random.seed(0)
    # attractor = clifford_attractor(-1.7, 1.3, -0.1, -1.21)
    attractor = clifford_attractor(-1.7, -1.3, -0.1, -1.21)
    AttractorPlotter(attractor).render_to("results/clifford_0.png")


if __name__ == "__main__":
    main()
