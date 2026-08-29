"""Execute both notebooks in place so charts and outputs ship with the file."""
import sys, time
import nbformat
from nbclient import NotebookClient

for path in (sys.argv[1:] or ["Hemophilia_Capstone_Clinical.ipynb",
                              "Hemophilia_Capstone_Accuracy.ipynb"]):
    t0 = time.time()
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    client = NotebookClient(nb, timeout=1800, kernel_name="python3",
                            resources={"metadata": {"path": "."}},
                            allow_errors=False)
    try:
        client.execute()
    except Exception as exc:
        print(f"FAILED {path}: {type(exc).__name__}: {str(exc)[:2000]}")
        raise
    nbformat.write(nb, path)
    n_out = sum(len(c.get("outputs", [])) for c in nb.cells)
    n_img = sum(1 for c in nb.cells for o in c.get("outputs", [])
                if "image/png" in o.get("data", {}))
    print(f"OK {path}: {len(nb.cells)} cells, {n_out} outputs, "
          f"{n_img} charts, {time.time()-t0:.0f}s")
