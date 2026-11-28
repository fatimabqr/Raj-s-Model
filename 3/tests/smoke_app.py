"""Executes app.py top-to-bottom against a stub `streamlit` (no server needed) to catch wiring errors.
Real Streamlit is still the final check:  streamlit run app.py"""
import sys, types, pathlib, runpy
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class Ctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, n): return lambda *a, **k: _default(n, a, k)
    def __iter__(self): return iter([Ctx(), Ctx(), Ctx(), Ctx(), Ctx(), Ctx(), Ctx()])


class State(dict):
    __getattr__ = dict.get
    def __setattr__(self, k, v): self[k] = v


def _default(name, a, k):
    if name == "radio" and a and a[0] == "Source": return a[1][1]          # choose Snapshot mode
    if name in ("radio", "selectbox"):
        return (a[1] if len(a) > 1 else k["options"])[k.get("index", 0)]
    if name in ("slider", "number_input"): return a[3] if len(a) > 3 and name == "slider" and False else (a[3] if len(a) > 3 else a[2])
    if name == "checkbox": return a[1] if len(a) > 1 else k.get("value", False)
    if name in ("date_input", "time_input"): return a[1] if len(a) > 1 else None
    if name == "text_input": return a[1] if len(a) > 1 else ""
    if name in ("button", "form_submit_button", "file_uploader"): return None if name == "file_uploader" else False
    if name == "columns": return [Ctx() for _ in range(a[0] if isinstance(a[0], int) else len(a[0]))]
    if name == "tabs": return [Ctx() for _ in a[0]]
    if name == "multiselect": return []
    if name == "data_editor": return a[0]
    if name == "cache_data": return (lambda f: f) if not a else a[0]
    if name == "progress": return Ctx()
    if name in ("form", "expander", "spinner"): return Ctx()
    return Ctx()


st = types.ModuleType("streamlit")
st.session_state = State()
st.sidebar = Ctx()
def cache_data(*a, **k): return (lambda f: f) if (not a or not callable(a[0])) else a[0]
st.cache_data = cache_data
def stop(): raise SystemExit
st.stop = stop; st.rerun = lambda: None
st.dataframe = lambda *a, **k: (a[0].style if hasattr(a[0], "style") else None)
for n in ("set_page_config title subheader markdown caption info success warning write metric bar_chart line_chart "
          "download_button json").split():
    setattr(st, n, lambda *a, **k: Ctx())
st.error = lambda *a, **k: print("ST.ERROR:", a)
for n in ("radio slider number_input checkbox date_input time_input text_input button form_submit_button file_uploader "
          "columns tabs multiselect data_editor progress form expander selectbox").split():
    setattr(st, n, (lambda nm: lambda *a, **k: _default(nm, a, k))(n))
st.__getattr__ = lambda n: (lambda *a, **k: _default(n, a, k))   # any other st.* call
sys.modules["streamlit"] = st
runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
print("app.py executed end-to-end with stub streamlit (snapshot mode)")
