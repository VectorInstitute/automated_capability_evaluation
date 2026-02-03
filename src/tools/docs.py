"""Documentation retrieval from local HTML files.

Structure-aware RAG that exploits naming conventions of HTML documentation
to deterministically build a Library -> Module -> Function index.
"""

import logging
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from bs4 import BeautifulSoup

log = logging.getLogger("tools.docs")


class ScientificDocRetriever:
    """
    Parses local HTML documentation directories to build a structured index
    of Libraries -> Modules -> Functions.
    """

    def __init__(self, docs_base_path: Path):
        self.docs_path = docs_base_path
        # Schema: index[library][module] = [list_of_function_names]
        self._index: Dict[str, Dict[str, List[str]]] = {} 
        # Schema: file_map[library.module.function] = Path(file.html)
        self._file_map: Dict[str, Path] = {}
        
        self._build_index()

    def _build_index(self):
        """Deterministically builds index based on 'reference/generated' file patterns."""
        
        # 1. Index NumPy and SciPy (Pattern: pkg.submodule.function.html)
        for lib in ["numpy", "scipy"]:
            # Find versioned directory (e.g., numpy-html-1.17.0)
            lib_dir = next(self.docs_path.glob(f"{lib}-html-*"), None)
            if not lib_dir: 
                log.warning(f"Documentation directory not found for {lib}")
                continue

            # Target specific reference directory to avoid tutorials/dev noise
            generated_dir = lib_dir / "reference" / "generated"
            if generated_dir.exists():
                if lib not in self._index:
                    self._index[lib] = defaultdict(list)
                    
                for html_file in generated_dir.glob("*.html"):
                    name = html_file.stem  # e.g., "numpy.linalg.eig"
                    
                    if name.startswith("index") or name.startswith("gallery"): 
                        continue
                    
                    parts = name.split('.')
                    if len(parts) >= 3:
                        # numpy.linalg.eig -> module="linalg", func="eig"
                        module = parts[1]
                        func = parts[-1]
                        
                        self._index[lib][module].append(func)
                        self._file_map[f"{lib}.{module}.{func}"] = html_file
                        
                log.info(f"Indexed {lib}: {len(self._index[lib])} modules")

        # 2. Index SymPy (Pattern: modules/topic/file.html)
        sympy_dir = next(self.docs_path.glob("sympy-docs-html-*"), None)
        if sympy_dir:
            modules_dir = sympy_dir / "modules"
            if modules_dir.exists():
                if "sympy" not in self._index: 
                    self._index["sympy"] = defaultdict(list)
                
                # SymPy organizes by folder names in 'modules/'
                for category_dir in modules_dir.iterdir():
                    if category_dir.is_dir() and not category_dir.name.startswith('_'):
                        topic = category_dir.name  # e.g., "matrices", "solvers"
                        for html_file in category_dir.glob("*.html"):
                            if html_file.stem != "index":
                                self._index["sympy"][topic].append(html_file.stem)
                                self._file_map[f"sympy.{topic}.{html_file.stem}"] = html_file
                                
                log.info(f"Indexed sympy: {len(self._index['sympy'])} modules")

    def get_library_overview(self) -> str:
        """Returns a high-level summary of available libraries and their modules."""
        overview = []
        for lib, modules in self._index.items():
            overview.append(f"### {lib.upper()}")
            # List modules, sorted alphabetically
            mod_list = sorted(modules.keys())
            overview.append(f"Available Modules: {', '.join(mod_list)}\n")
        return "\n".join(overview)

    def get_full_module_context(self, library: str, module: str) -> str:
        """
        Retrieves signatures for ALL functions in a module.
        Intended for high-context models (Gemini Flash).
        """
        if library not in self._index or module not in self._index[library]:
            return f"Error: Module {library}.{module} not found."

        # SORT the functions so the prompt is deterministic
        functions = sorted(self._index[library][module])
        context_blocks = [f"--- DOCUMENTATION FOR {library}.{module} ---"]
        
        for func in functions:
            key = f"{library}.{module}.{func}"
            file_path = self._file_map.get(key)
            if file_path:
                signature = self._extract_signature_from_html(file_path)
                if signature:
                    context_blocks.append(signature)
        
        return "\n\n".join(context_blocks)

    def _extract_signature_from_html(self, path: Path) -> str:
        """Parses HTML to extract signature + brief description (Optimized for Tokens)."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                
                extracted_content = []

                # Find Main Definitions
                definitions = soup.find_all('dl', class_=['function', 'class', 'method'])
                
                for dl in definitions:
                    # 1. Get the signature (dt)
                    dt = dl.find('dt')
                    if not dt: 
                        continue
                    
                    # CLEANUP: Remove ¶, [source], and extra whitespace
                    raw_sig = dt.get_text().replace("¶", "").replace("[source]", "").strip()
                    sig = " ".join(raw_sig.split())
                    
                    # FILTER 1: Skip private/magic methods in the main definition
                    # (e.g. skip scipy.sparse.bsr_matrix.__add__)
                    func_name = sig.split('(')[0].split('.')[-1]
                    if func_name.startswith('_'): 
                        continue

                    # 2. Get the description (dd) - First sentence only, max 120 chars
                    desc_text = ""
                    dd = dl.find('dd')
                    if dd:
                        p = dd.find('p')
                        if p:
                            raw_desc = " ".join(p.get_text().split())
                            # Split by period to get first sentence, or cap at 120 chars
                            if '.' in raw_desc:
                                desc_text = raw_desc.split('.')[0] + "."
                            else:
                                desc_text = raw_desc
                            
                            if len(desc_text) > 120:
                                desc_text = desc_text[:117] + "..."
                    
                    # FORMAT: Dense one-liner
                    # e.g., "numpy.linalg.eig(a): Compute eigenvalues."
                    entry = f"{sig}: {desc_text}" if desc_text else sig
                    extracted_content.append(entry)

                    # 3. SYMPY/CLASS HANDLING: Look for methods inside
                    # Only do this if it's a class definition
                    if "class" in dl.get('class', []) or "sympy" in str(path):
                        methods = dl.find_all('dl', class_='method')
                        for method in methods:
                            m_dt = method.find('dt')
                            if m_dt:
                                m_sig_raw = m_dt.get_text().replace("¶", "").replace("[source]", "").strip()
                                m_sig = " ".join(m_sig_raw.split())
                                
                                # FILTER 2: Strict method filtering
                                # Skip __init__, __add__, _private_method
                                method_name = m_sig.split('(')[0].split('.')[-1]
                                if method_name.startswith('_'):
                                    continue
                                
                                # Optional: Skip simple properties to save space?
                                # if "property" in str(method): continue

                                extracted_content.append(f"  . {method_name}(...): Method") 
                                # Note: We purposefully omit method descriptions to save tokens
                                # relying on the method name being descriptive enough.

                if extracted_content:
                    return "\n".join(extracted_content)

        except Exception as e:
            # log.warning(f"Error parsing {path}: {e}")
            pass
        return ""
