import argparse


REQUIRED_OUTPUT_FIELDS = [
    "id",
    "text",
    "head_entity",
    "head_type",
    "tail_entity",
    "tail_type",
    "gold_relation",
    "split",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert DDI 2013 into unified Stage 1 JSONL. "
            "This placeholder documents the target contract; wire XML parsing after data is available."
        )
    )
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    raise SystemExit(
        "DDI 2013 raw XML files are not present yet. Expected output fields: "
        + ", ".join(REQUIRED_OUTPUT_FIELDS)
        + f". raw_dir={args.raw_dir} output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
