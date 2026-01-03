import time
import traceback

print("[TEST] Importing time_graph_cpp...")
try:
    import time_graph_cpp as tgcpp
    print("  OK, version:", getattr(tgcpp, "__version__", "unknown"))
except Exception as e:
    print("  IMPORT ERROR:", e)
    traceback.print_exc()
    raise SystemExit(1)

mpai_path = r"..\\time_graph_x\\hazir\\test_data.mpai"
print(f"[TEST] Opening MPAI: {mpai_path}")
try:
    reader = tgcpp.MpaiReader(mpai_path)
    print("  rows=", reader.get_row_count(), "cols=", reader.get_column_count())
    cols = reader.get_column_names()
    print("  columns:", cols)
except Exception as e:
    print("  MPAI OPEN ERROR:", e)
    traceback.print_exc()
    raise SystemExit(1)

if not cols:
    print("  No columns in MPAI file")
    raise SystemExit(1)

# Detect time column
time_col = None
for c in cols:
    if "time" in c.lower():
        time_col = c
        break
if time_col is None:
    time_col = cols[0]

value_col = None
for c in cols:
    if c != time_col:
        value_col = c
        break

print("  time_col=", time_col, " value_col=", value_col)

cond_list = []
if value_col is not None:
    cond = tgcpp.FilterCondition()
    cond.column_name = value_col
    cond.type = tgcpp.FilterType.RANGE
    cond.min_value = -1e9
    cond.max_value = 1e9
    cond.threshold = 0.0
    cond.op = tgcpp.FilterOperator.AND
    cond_list.append(cond)

engine = tgcpp.FilterEngine()
print("[TEST] Running calculate_streaming() ...")
start = time.perf_counter()
segments = engine.calculate_streaming(reader, time_col, cond_list, 0, 0)
elapsed = time.perf_counter() - start
print("  segments count:", len(segments))
if segments:
    first = segments[0]
    print(
        "  first segment:",
        first.start_time,
        first.end_time,
        first.start_index,
        first.end_index,
    )
print(f"  elapsed: {elapsed*1000:.2f} ms")


