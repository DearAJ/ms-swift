#!/usr/bin/env python3
"""Validate the local Agent SFT JSONL before distributed training."""

import argparse
import json
from collections import Counter
from pathlib import Path


VALID_ROLES = {'system', 'user', 'assistant', 'tool_call', 'tool', 'tool_response'}


def validate(path: Path) -> None:
    counts = Counter()
    invalid = []
    total_bytes = 0

    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                invalid.append(f'line {line_number}: empty line')
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                invalid.append(f'line {line_number}: invalid JSON ({error.msg})')
                continue

            messages = item.get('messages')
            if not isinstance(messages, list) or not messages:
                invalid.append(f'line {line_number}: missing non-empty messages list')
                continue

            for message in messages:
                role = message.get('role') if isinstance(message, dict) else None
                content = message.get('content') if isinstance(message, dict) else None
                if role not in VALID_ROLES:
                    invalid.append(f'line {line_number}: unsupported role {role!r}')
                    break
                if not isinstance(content, (str, list, type(None))):
                    invalid.append(f'line {line_number}: invalid content type for {role!r}')
                    break
                counts[role] += 1
                if isinstance(content, str):
                    total_bytes += len(content.encode('utf-8'))

    print(f'dataset: {path}')
    print(f'samples: {line_number}')
    print(f'message roles: {dict(sorted(counts.items()))}')
    print(f'message UTF-8 bytes: {total_bytes}')
    if invalid:
        print('validation failed:')
        for error in invalid[:20]:
            print(f'  {error}')
        if len(invalid) > 20:
            print(f'  ... and {len(invalid) - 20} more errors')
        raise SystemExit(1)
    print('validation: passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset',
        nargs='?',
        type=Path,
        default=Path('/home/accio_data/workgroup/aijunyang/oss_data/tb2_sft.jsonl'))
    args = parser.parse_args()
    validate(args.dataset)
