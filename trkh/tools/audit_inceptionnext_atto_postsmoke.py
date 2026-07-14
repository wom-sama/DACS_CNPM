from __future__ import annotations

from trkh.tools.audit_tokenizer_postsmoke import (
    assess_postsmoke,
    main,
    parse_args,
    run_audit,
)

__all__ = ("assess_postsmoke", "main", "parse_args", "run_audit")


if __name__ == "__main__":
    raise SystemExit(main())
