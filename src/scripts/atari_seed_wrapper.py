import sys
import time
from pathlib import Path
from typing import Any, Union

import ale_py
import ale_py.roms
import gym

SUPPORTED_GAMES = ["Riverraid"]


def patch_atari_seed(
    ale_py: Any, seed: Union[int, str], game: str = "Riverraid"
) -> Any:
    """Patches an atari ROM with a new random seed.

    Parameters
    ----------
    ale_py : Any
        ale_py module instance.
    seed : Union[int, str]
        Environment seed to be patched.
    game : str, optional
        ale_py ROM name, by default "Riverraid"

    Returns
    -------
    Any
        Patches ale_py module.

    """
    if game not in SUPPORTED_GAMES:
        raise ValueError(
            f"{game} is not currently supported. Please choose one from: {', '.join(SUPPORTED_GAMES)}"
        )

    seed = int(seed) % 2 ** 16
    seed_bytes = seed.to_bytes(2, "little")  # seed_lo, seed_hi

    rom_name = f"{game.lower()}.bin"
    roms_path = Path(ale_py.__file__).parent / "roms"
    source_rom_path = roms_path / rom_name
    target_rom_path = Path("/tmp") / rom_name

    with open(source_rom_path, "rb") as source_rom_file:
        rom = source_rom_file.read()

    rom = bytearray(rom)

    if rom[0xDC9 : 0xDC9 + 2] != b"\x14\xA8" or rom[0xDCE : 0xDCE + 2] != b"\x14\xA8":
        print(
            f"incompatible values found in source ROM {source_rom_path}:",
            rom[0xDC9 : 0xDC9 + 2],
            rom[0xDCE : 0xDCE + 2],
        )
        exit(1)

    # Player 1
    rom[0xDC9 : 0xDC9 + 2] = seed_bytes
    # Player 2
    rom[0xDCE : 0xDCE + 2] = seed_bytes

    with open(target_rom_path, "wb") as target_rom_file:
        target_rom_file.write(rom)

    print(f"wrote file {target_rom_path} with random seed {seed}")

    # --- Modifies ALE environment to point to patched ROM
    ale_py.roms.Riverraid = Path(target_rom_path).resolve()

    return ale_py


if __name__ == "__main__":

    seed = sys.argv[1]
    ale_py = patch_atari_seed(ale_py, seed, "Riverraid")

    # --- Creates and simulates environment with patched ROM
    NUM_STEPS = 500
    env = gym.make("Riverraid", render_mode="human")
    env.reset(seed=seed)
    for step in range(NUM_STEPS):
        # samples random action
        action = env.action_space.sample()

        # apply the action
        obs, reward, done, info = env.step(action)

        # Wait a bit before the next frame unless you want to see a crazy fast video
        time.sleep(0.001)

        # If the epsiode is up, then start another one
        if done:
            env.reset()

    # Close the env
    env.close()
