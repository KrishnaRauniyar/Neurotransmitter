import argparse

def convert_group(name):
    """
    Takes a string like '1D0K_A_404_DGL' and returns:
    '1D0K_A_404_DGL_GLU'
    """
    parts = name.split("_")

    # Group is at index 3
    group = parts[3]

    # Map DGL → GLU, otherwise keep same
    final_group = "GLU" if group == "DGL" else group

    return f"{name}_{final_group}"


def process_file(input_file, output_file):
    with open(input_file, "r") as f:
        lines = f.read().strip().split("\n")

    output_lines = []
    for line in lines:
        left, values = line.split(";")
        new_left = convert_group(left)
        output_lines.append(f"{new_left};{values}")

    with open(output_file, "w") as f:
        f.write("\n".join(output_lines))

    print(f"[✓] Output saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Append group name to residue ID")

    parser.add_argument("--input", "-i", required=True,
                        help="Input CSV file")
    parser.add_argument("--output", "-o", required=True,
                        help="Output CSV file")

    args = parser.parse_args()

    process_file(args.input, args.output)
