#!/usr/bin/env python3.11
"""Cache functions in the file system."""
import pathlib as p
import pickle
import hashlib as hl


def fscache(folder: p.Path | str, hasher=lambda *a, **kw: (a, kw)) -> callable:
    folder = p.Path(folder)

    def decorator(function: callable) -> callable:
        def cached_func(*a, **kw):
            hash_str = str(hasher(*a, **kw))
            hashed = hl.md5(hash_str.encode("utf-8")).hexdigest()

            hash_path = folder / function.__name__ / f"{hashed}.data"

            if not hash_path.parent.exists():
                hash_path.parent.mkdir()

            if hash_path.exists():
                cached_func.hits += 1

                with hash_path.open("rb") as file:
                    return pickle.load(file)

            result = function(*a, **kw)
            cached_func.misses += 1

            with hash_path.open("wb") as file:
                pickle.dump(result, file)

            return result

        def clear():
            """Clear the cache."""
            for file in folder.iterdir():
                file.unlink()

        cached_func.clear = clear
        cached_func.hits = 0
        cached_func.misses = 0

        return cached_func

    return decorator
