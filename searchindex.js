Search.setIndex({"docnames": ["howto/eam-forces", "howto/force-field-training", "howto/index", "howto/property-regression", "howto/test", "index", "overview", "reference/api", "reference/generated/nfflr.Atoms", "reference/generated/nfflr.AtomsDataset", "reference/generated/nfflr.models.ALIGNN", "reference/generated/nfflr.models.SchNet", "reference/generated/nfflr.models.Tersoff", "reference/generated/nfflr.nn.FeedForward", "reference/generated/nfflr.nn.MLPLayer", "reference/generated/nfflr.nn.PeriodicAdaptiveRadiusGraph", "reference/generated/nfflr.nn.PeriodicKShellGraph", "reference/generated/nfflr.nn.PeriodicRadiusGraph", "reference/index", "reference/models", "reference/nn", "tutorials/atoms", "tutorials/index", "tutorials/quickstart"], "filenames": ["howto/eam-forces.ipynb", "howto/force-field-training.ipynb", "howto/index.md", "howto/property-regression.ipynb", "howto/test.ipynb", "index.md", "overview.md", "reference/api.md", "reference/generated/nfflr.Atoms.rst", "reference/generated/nfflr.AtomsDataset.rst", "reference/generated/nfflr.models.ALIGNN.rst", "reference/generated/nfflr.models.SchNet.rst", "reference/generated/nfflr.models.Tersoff.rst", "reference/generated/nfflr.nn.FeedForward.rst", "reference/generated/nfflr.nn.MLPLayer.rst", "reference/generated/nfflr.nn.PeriodicAdaptiveRadiusGraph.rst", "reference/generated/nfflr.nn.PeriodicKShellGraph.rst", "reference/generated/nfflr.nn.PeriodicRadiusGraph.rst", "reference/index.md", "reference/models.md", "reference/nn.md", "tutorials/atoms.ipynb", "tutorials/index.md", "tutorials/quickstart.ipynb"], "titles": ["Checking autograd forces against analytical forces for Embedded Atom model", "Property regression example", "How-to guides", "Property regression example", "an example", "NFFLr documentation", "Overview", "nfflr", "Atoms", "AtomsDataset", "ALIGNN", "SchNet", "Tersoff", "FeedForward", "MLPLayer", "nfflr.nn.PeriodicAdaptiveRadiusGraph", "nfflr.nn.PeriodicKShellGraph", "nfflr.nn.PeriodicRadiusGraph", "Reference", "nfflr.models", "nfflr.nn", "Atoms", "Tutorials", "Quickstart"], "terms": {"thi": [0, 1, 3, 4, 6, 7, 8, 23], "notebook": 0, "compar": 0, "comput": [0, 23], "automat": [0, 1, 3, 23], "differenti": [0, 23], "respect": [0, 23], "bond": [0, 10, 12, 23], "vector": [0, 6, 12, 23], "coordin": [0, 8, 23], "al99": 0, "implement": [0, 6, 12, 23], "ASE": 0, "first": 0, "fetch": 0, "eam": 0, "data": [0, 1, 3, 4, 6, 8, 10, 21, 22], "from": [0, 1, 3, 6, 21, 23], "interatom": [0, 18], "repositori": 0, "curl": 0, "http": [0, 1, 3, 12, 23], "www": [0, 3, 12, 23], "ctcm": [0, 12], "nist": [0, 5, 12], "gov": [0, 5, 12], "download": 0, "1999": 0, "mishin": 0, "y": [0, 1, 3, 10], "farka": 0, "d": [0, 4, 12, 23], "mehl": 0, "m": [0, 12], "j": [0, 12], "papaconstantopoulo": 0, "A": [0, 6, 12], "al": 0, "2": [0, 1, 3, 4, 6, 9, 12, 21, 23], "alloi": 0, "o": [0, 6], "total": [0, 6, 9, 23], "receiv": 0, "xferd": 0, "averag": 0, "speed": 0, "time": [0, 23], "current": [0, 4, 6, 23], "dload": 0, "upload": 0, "spent": 0, "left": 0, "0": [0, 1, 3, 4, 6, 9, 12, 15, 16, 17, 21, 23], "100": [0, 1, 3, 4, 6], "762k": 0, "3474k": 0, "3480k": 0, "gradient": [0, 23], "i": [0, 1, 3, 4, 5, 6, 8, 9, 12, 23], "best": 0, "done": 0, "64": [0, 10], "bit": 0, "float": [0, 9, 12, 15, 16, 17], "point": [0, 23], "precis": 0, "plum": [0, 1, 23], "import": [0, 1, 3, 4, 6, 21, 23], "activate_autoreload": 0, "pathlib": [0, 6], "path": [0, 6, 9], "torch": [0, 1, 3, 4, 6, 15, 16, 17, 21, 23], "ase": [0, 21, 23], "calcul": [0, 9], "build": 0, "bulk": 0, "nfflr": [0, 2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 18, 21, 23], "graph": [0, 1, 3, 4, 6, 9, 10, 12, 17, 18, 23], "classic": [0, 4, 18], "torcheam": 0, "set_default_dtyp": [0, 4], "float64": [0, 4], "modulenotfounderror": 0, "traceback": [0, 1, 3, 4, 23], "most": [0, 1, 3, 4, 23], "recent": [0, 1, 3, 4, 23], "call": [0, 1, 3, 4, 23], "last": [0, 1, 3, 4, 23], "cell": [0, 1, 3, 4, 8, 21, 23], "In": [0, 1, 3, 4, 23], "line": [0, 1, 3, 4, 6, 10, 23], "12": [0, 3, 4, 16], "10": [0, 1, 3, 4, 12, 23], "11": [0, 1, 3, 4], "14": [0, 1, 3, 4, 23], "file": [0, 1, 3, 4, 5, 6], "work": [0, 1, 3, 4, 5, 8], "py": [0, 1, 3, 4, 6], "9": [0, 1, 3, 4, 23], "nn": [0, 1, 3, 4, 6, 13, 14, 18], "torchcubicsplin": 0, "natural_cubic_spline_coeff": 0, "naturalcubicsplin": 0, "17": [0, 23], "dataclass": [0, 4], "18": 0, "class": [0, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], "eamdata": 0, "No": 0, "modul": [0, 1, 4, 10, 11, 23], "name": [0, 1, 3, 4, 6, 23], "we": [0, 1, 3, 4, 23], "set": [0, 1, 3, 6, 23], "up": [0, 1, 3, 6], "fcc": 0, "aluminum": 0, "add": 0, "amount": 0, "jitter": 0, "4": [0, 1, 3, 4, 6, 10, 11, 12, 21, 23], "05": [0, 1, 6], "al_as": 0, "rattl": 0, "stdev": 0, "wrap": 0, "ase_eam": 0, "set_calcul": 0, "pytorch": [0, 1, 3, 4, 23], "version": [0, 3, 4, 23], "get_cel": 0, "arrai": [0, 4, 23], "get_scaled_posit": 0, "number": [0, 8, 21, 23], "torch_eam": 0, "dtype": [0, 1, 3, 4, 15, 16, 17, 23], "The": [0, 5, 6, 23], "us": [0, 2, 4, 5, 6, 23], "spline": 0, "compon": [0, 1, 3, 5, 23], "while": 0, "energi": [0, 1, 3, 4, 9, 23], "displac": 0, "sum": 0, "reduct": 0, "aggreg": 0, "individu": 0, "both": [0, 6, 23], "all": [0, 4, 23], "match": [0, 4], "within": 0, "construct": [0, 6, 18], "radiu": [0, 17], "cutoff": [0, 1, 3, 4, 6, 10, 11, 12, 15, 16, 17], "g": [0, 1, 3, 4, 6, 10, 12, 23], "periodic_radius_graph": [0, 1, 3, 4, 6, 23], "r": [0, 1, 3, 4, 6, 12, 23], "evalu": [0, 12], "e_dgl": 0, "force_dgl": 0, "detach": [0, 4], "e_as": 0, "get_potential_energi": 0, "force_as": 0, "get_forc": 0, "print": [0, 23], "f": [0, 4, 23], "item": [0, 1, 3], "np": [0, 1, 3, 4], "isclos": 0, "user": [0, 3, 4], "bld": [0, 3, 4, 6], "pyenv": [0, 3, 4], "3": [0, 1, 3, 4, 6, 12, 21, 23], "env": [0, 3, 4], "lib": [0, 1, 3, 4], "python3": [0, 1, 3, 4], "site": [0, 1, 3, 4, 8], "packag": [0, 1, 3, 4], "dgl": [0, 3, 4, 6, 9, 10, 12], "backend": [0, 3, 4, 5, 6], "tensor": [0, 1, 3, 4, 6, 8, 9, 10, 11, 12, 13, 21, 23], "445": [0, 3, 4], "userwarn": [0, 3, 4], "typedstorag": [0, 3, 4], "deprec": [0, 3, 4], "It": [0, 3, 4], "remov": [0, 3, 4], "futur": [0, 3, 4], "untypedstorag": [0, 3, 4], "onli": [0, 3, 4], "storag": [0, 3, 4], "should": [0, 3, 4, 6], "matter": [0, 3, 4], "you": [0, 3, 4], "ar": [0, 3, 4, 5, 6, 9, 23], "directli": [0, 3, 4, 23], "To": [0, 1, 3, 4, 6], "access": [0, 3, 4], "untyped_storag": [0, 3, 4], "instead": [0, 3, 4], "assert": [0, 3, 4], "input": [0, 3, 4, 5], "numel": [0, 3, 4], "size": [0, 1, 3, 4, 6], "214": 0, "15885773313198": 0, "15885773313158": 0, "true": [0, 1, 3, 4, 6, 12, 23], "section": 0, "perform": [0, 5], "same": [0, 23], "diagnost": 0, "lammp": [0, 12], "demonstr": 0, "rel": [0, 12], "posit": [0, 8, 21, 23], "reduc": 0, "correct": 0, "paramet": [0, 1, 3, 4, 8, 12], "correspond": [0, 12], "entri": [0, 12], "1988": [0, 12], "si": [0, 12], "b": [0, 12], "1": [0, 1, 3, 4, 6, 9, 10, 11, 12, 21, 23], "1988_si": 0, "cat": 0, "532": 0, "1299": 0, "1310": 0, "citat": 0, "phy": [0, 12], "rev": [0, 12], "37": [0, 12], "6991": [0, 12], "valu": [0, 1, 12, 23], "verifi": 0, "luca": 0, "hale": 0, "ident": [0, 8], "august": 0, "22": [0, 12, 21, 23], "2018": 0, "distribut": [0, 1, 3, 6], "openkim": 0, "mo_245095684871_001": 0, "metal": 0, "unit": 0, "e1": [0, 12], "e2": [0, 12], "e3": [0, 12], "gamma": [0, 12], "lambda3": [0, 12], "c": [0, 4, 12], "costheta0": [0, 12], "n": [0, 12, 23], "beta": [0, 12], "lambda2": [0, 12], "lambda1": [0, 12], "3258": [0, 12], "8381": [0, 12], "0417": [0, 12], "956": [0, 12], "33675": [0, 12], "95": [0, 4, 12], "373": [0, 12], "2394": [0, 12], "3264": [0, 12], "7": [0, 1, 3, 4, 12, 23], "seem": 0, "still": 0, "have": [0, 1, 4, 6, 23], "issu": [0, 5], "pars": 0, "thermo": 0, "log": [0, 4], "can": [0, 1, 3, 4, 5, 6, 23], "around": 0, "short": 0, "term": [0, 12], "monkei": 0, "patch": 0, "read_lammps_log": 0, "follow": [0, 12], "discuss": [0, 5], "here": [0, 4, 23], "io": 0, "lammpsrun": 0, "parallel": [0, 6], "paropen": 0, "re": [0, 4], "compil": 0, "re_compil": 0, "ignorecas": 0, "calculation_end_mark": 0, "def": [0, 1, 4], "self": [0, 1, 3, 10], "lammps_log": 0, "none": [0, 1, 4, 8, 9, 10, 11, 13, 21, 23], "thermo_arg": 0, "calc": 0, "base": [0, 2], "gitlab": 0, "com": [0, 1, 3, 23], "1096": 0, "method": [0, 1, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], "which": [0, 4, 23], "read": [0, 5], "output": 0, "label": [0, 4], "isinst": [0, 3], "str": [0, 6, 9, 10, 11, 12], "fileobj": 0, "rb": 0, "close_log_fil": 0, "els": [0, 1, 4], "expect": [0, 23], "lammps_in": 0, "like": [0, 6, 23], "object": [0, 1, 3], "fals": [0, 1, 3, 4, 6, 9, 10, 11], "read_log": 0, "depend": [0, 6], "three": 0, "thermo_styl": 0, "custom": [0, 6], "arg": [0, 1], "capit": 0, "e": [0, 3, 6, 23], "don": [0, 1, 3], "t": [0, 1, 3, 4], "ke": 0, "cpu": [0, 4], "kineng": 0, "mark_r": 0, "": [0, 1, 3, 6], "join": 0, "x": [0, 4, 6, 10, 13, 14], "_custom_thermo_mark": 0, "todo": 0, "regex": 0, "magic": 0, "necessari": 0, "someth": 0, "convert": [0, 5, 23], "f_re": 0, "nan": 0, "inf": 0, "n_arg": 0, "len": [0, 1], "creat": 0, "exactli": 0, "white": 0, "space": 0, "separ": 0, "floatish": 0, "thing": 0, "_custom_thermo_r": 0, "flag": 0, "thermo_cont": 0, "readlin": 0, "decod": 0, "utf": 0, "8": [0, 1, 3, 4, 6, 9, 23], "strip": 0, "error": [0, 4], "close": 0, "rais": [0, 4], "runtimeerror": 0, "exit": 0, "messag": 0, "get": [0, 9, 23], "bool_match": 0, "dictionari": [0, 23], "between": 0, "each": [0, 12, 23], "append": [0, 1, 3, 4], "dict": [0, 6, 9], "zip": 0, "map": 0, "group": 0, "return": [0, 1, 3, 4, 6, 23], "parser": 0, "now": [0, 1, 3], "serv": 0, "refer": [0, 1, 3, 12, 23], "exampl": [0, 2, 23], "crystal": [0, 8, 9, 10], "si_as": 0, "diamond": 0, "5": [0, 1, 3, 4, 6, 15, 17, 21, 23], "43": [0, 3], "01": [0, 1, 3, 4, 23], "seed": 0, "36": 0, "configur": [0, 1, 3, 5, 6, 23], "command": 0, "opt": [0, 1, 4], "homebrew": 0, "bin": [0, 4], "lmp_serial": 0, "binary_dump": 0, "updat": [0, 10], "pair_styl": 0, "pair_coeff": 0, "e_lammp": 0, "f_lammp": 0, "592": 0, "3985671491665": 0, "20522987": 0, "26851818": 0, "14179856": 0, "06219078": 0, "05221548": 0, "0728464": 0, "03842853": 0, "02849059": 0, "04544221": 0, "21198316": 0, "12308603": 0, "00683117": 0, "5269759": 0, "12608296": 0, "29867076": 0, "And": 0, "tersoffconfig": [0, 12], "out": 0, "e_tersoff": 0, "total_energi": [0, 4, 6, 23], "f_tersoff": 0, "stress_tersoff": 0, "stress": [0, 1, 6, 9, 23], "get_volum": 0, "3986": 0, "2052": 0, "2685": 0, "1418": 0, "0622": 0, "0522": 0, "0728": 0, "0384": 0, "0285": 0, "0454": 0, "2120": 0, "1231": 0, "0068": 0, "5270": 0, "1261": 0, "2987": 0, "6": [0, 1, 3, 4, 6, 23], "5235e": 0, "04": 0, "6092e": 0, "03": [0, 1, 3, 23], "2400e": 0, "5744e": 0, "0972e": 0, "06": [0, 1, 3, 23], "4832e": 0, "numer": 0, "so": [0, 8, 23], "do": 0, "gradcheck": 0, "atol": 0, "1e": [0, 1, 3, 4, 6], "rtol": 0, "001": 0, "largest": 0, "discrep": 0, "about": [0, 8], "ev": [0, 4, 6, 9, 10, 11], "mathrm": [0, 4], "aa": [0, 4], "numpi": [0, 1, 3], "max": [0, 23], "749736085540081e": 0, "07": [0, 1, 3, 23], "also": [0, 6, 23], "get_stress": 0, "52352915e": 0, "57444097e": 0, "48324331e": 0, "09715783e": 0, "23995521e": 0, "60922348e": 0, "voigt_6_to_full_3x3_stress": 0, "show": [1, 3, 4], "how": [1, 3], "togeth": [1, 3], "let": [1, 3], "train": [1, 3, 4, 5, 9, 22], "format": [1, 3, 4, 5, 6, 8, 23], "model": [1, 2, 3, 4, 5, 10, 11, 12, 18, 22], "dft_3d": [1, 3, 4, 23], "dataset": [1, 3, 4, 6, 9, 22], "transform": [1, 3, 4, 6, 8, 9, 10, 11, 17, 23], "atomsdataset": [1, 3, 4, 6, 18], "atom": [1, 2, 3, 4, 6, 9, 10, 11, 18, 22], "dglgraph": [1, 3, 6, 8, 9, 10, 12, 23], "periodicradiusgraph": [1, 3, 10, 11, 18], "mlearn": [1, 4, 23], "target": [1, 3, 4, 6, 9, 23], "energy_and_forc": [1, 4, 6, 23], "dataset_nam": [1, 3, 23], "obtain": [1, 3, 23], "1730": [1, 23], "github": [1, 5, 23], "materialsvirtuallab": [1, 23], "00": [1, 3, 23], "57m": 1, "ib": [1, 3], "60": 1, "4k": 1, "443kib": 1, "269k": 1, "02": [1, 3, 23], "06mib": 1, "45": [1, 3, 4], "16m": 1, "45mib": 1, "20mib": 1, "load": [1, 3, 23], "zipfil": [1, 3, 23], "complet": [1, 3, 23], "num_nod": [1, 3, 4, 23], "107": 1, "num_edg": [1, 3, 23], "20940": 1, "ndata_schem": [1, 3, 23], "xfrac": [1, 3, 23], "scheme": [1, 3, 23], "shape": [1, 3, 23], "float32": [1, 3, 15, 16, 17, 23], "coord": [1, 3, 23], "atomic_numb": [1, 3, 23], "int32": [1, 3, 23], "edata_schem": [1, 3, 23], "64656": [1, 23], "0625": [1, 23], "forc": [1, 2, 4, 5, 6, 9, 22], "9282e": [1, 23], "8793e": [1, 23], "6374e": [1, 23], "2543e": [1, 23], "0313e": [1, 23], "6808e": [1, 23], "5372e": [1, 23], "4736e": [1, 23], "2997e": [1, 23], "5678e": [1, 23], "1175e": [1, 23], "0934e": [1, 23], "6499e": [1, 23], "6259e": [1, 23], "5255e": [1, 23], "6698e": [1, 23], "8080e": [1, 23], "7749e": [1, 23], "6802e": [1, 23], "1423e": [1, 23], "0166e": [1, 23], "0730e": [1, 23], "5780e": [1, 23], "1357e": [1, 23], "9132e": [1, 23], "1381e": [1, 23], "4296e": [1, 23], "0090e": [1, 23], "5143e": [1, 23], "5578e": [1, 23], "7128e": [1, 23], "7808e": [1, 23], "4215e": [1, 23], "3987e": [1, 23], "6757e": [1, 23], "9322e": [1, 23], "7190e": [1, 23], "0627e": [1, 23], "2933e": [1, 23], "6458e": [1, 23], "6833e": [1, 23], "0043e": [1, 23], "5756e": [1, 23], "5868e": [1, 23], "7038e": [1, 23], "2044e": [1, 23], "3979e": [1, 23], "5036e": [1, 23], "5743e": [1, 23], "4479e": [1, 23], "7272e": [1, 23], "8223e": [1, 23], "5903e": [1, 23], "7198e": [1, 23], "9518e": [1, 23], "7982e": [1, 23], "6208e": [1, 23], "3000e": [1, 23], "7643e": [1, 23], "0947e": [1, 23], "3517e": [1, 23], "4522e": [1, 23], "6359e": [1, 23], "4930e": [1, 23], "1648e": [1, 23], "1246e": [1, 23], "8361e": [1, 23], "0337e": [1, 23], "0099e": [1, 23], "4334e": [1, 23], "4563e": [1, 23], "8775e": [1, 23], "2193e": [1, 23], "8368e": [1, 23], "7678e": [1, 23], "8822e": [1, 23], "3724e": [1, 23], "0373e": [1, 23], "7925e": [1, 23], "4629e": [1, 23], "7126e": [1, 23], "3972e": [1, 23], "1936e": [1, 23], "4154e": [1, 23], "0657e": [1, 23], "6893e": [1, 23], "3909e": [1, 23], "2667e": [1, 23], "9585e": [1, 23], "0468e": [1, 23], "3723e": [1, 23], "7657e": [1, 23], "4826e": [1, 23], "3950e": [1, 23], "1809e": [1, 23], "7236e": [1, 23], "0571e": [1, 23], "0909e": [1, 23], "3469e": [1, 23], "2798e": [1, 23], "3690e": [1, 23], "8363e": [1, 23], "3372e": [1, 23], "8005e": [1, 23], "0848e": [1, 23], "7622e": [1, 23], "1141e": [1, 23], "8884e": [1, 23], "1697e": [1, 23], "0889e": [1, 23], "3894e": [1, 23], "1740e": [1, 23], "2013e": [1, 23], "5727e": [1, 23], "5217e": [1, 23], "6934e": [1, 23], "8191e": [1, 23], "4829e": [1, 23], "2664e": [1, 23], "1411e": [1, 23], "2328e": [1, 23], "2866e": [1, 23], "1776e": [1, 23], "2366e": [1, 23], "5056e": [1, 23], "3455e": [1, 23], "8714e": [1, 23], "4488e": [1, 23], "2792e": [1, 23], "0664e": [1, 23], "4243e": [1, 23], "2686e": [1, 23], "3897e": [1, 23], "7333e": [1, 23], "4011e": [1, 23], "0459e": [1, 23], "1634e": [1, 23], "0630e": [1, 23], "9009e": [1, 23], "2214e": [1, 23], "4072e": [1, 23], "3802e": [1, 23], "1611e": [1, 23], "3336e": [1, 23], "2308e": [1, 23], "7998e": [1, 23], "0719e": [1, 23], "5169e": [1, 23], "4886e": [1, 23], "4431e": [1, 23], "3966e": [1, 23], "3065e": [1, 23], "9503e": [1, 23], "8711e": [1, 23], "6996e": [1, 23], "6954e": [1, 23], "0038e": [1, 23], "8048e": [1, 23], "6736e": [1, 23], "8896e": [1, 23], "9839e": [1, 23], "1865e": [1, 23], "0303e": [1, 23], "5889e": [1, 23], "0517e": [1, 23], "4835e": [1, 23], "5193e": [1, 23], "8107e": [1, 23], "3507e": [1, 23], "6680e": [1, 23], "6512e": [1, 23], "6324e": [1, 23], "0497e": [1, 23], "7391e": [1, 23], "7163e": [1, 23], "8480e": [1, 23], "0546e": [1, 23], "5508e": [1, 23], "4519e": [1, 23], "3183e": [1, 23], "4062e": [1, 23], "8017e": [1, 23], "4209e": [1, 23], "2076e": [1, 23], "1055e": [1, 23], "7652e": [1, 23], "7866e": [1, 23], "0725e": [1, 23], "5774e": [1, 23], "6219e": [1, 23], "1061e": [1, 23], "6820e": [1, 23], "5689e": [1, 23], "1297e": [1, 23], "2079e": [1, 23], "3750e": [1, 23], "6904e": [1, 23], "2430e": [1, 23], "5449e": [1, 23], "4885e": [1, 23], "6164e": [1, 23], "6403e": [1, 23], "3929e": [1, 23], "3473e": [1, 23], "0026e": [1, 23], "1965e": [1, 23], "8875e": [1, 23], "0416e": [1, 23], "0578e": [1, 23], "4767e": [1, 23], "7263e": [1, 23], "0396e": [1, 23], "0797e": [1, 23], "2834e": [1, 23], "0441e": [1, 23], "1592e": [1, 23], "0053e": [1, 23], "6651e": [1, 23], "4538e": [1, 23], "1315e": [1, 23], "5051e": [1, 23], "6349e": [1, 23], "9915e": [1, 23], "2209e": [1, 23], "3324e": [1, 23], "9588e": [1, 23], "1156e": [1, 23], "3736e": [1, 23], "2689e": [1, 23], "6983e": [1, 23], "8699e": [1, 23], "6415e": [1, 23], "2089e": [1, 23], "2056e": [1, 23], "2394e": [1, 23], "3969e": [1, 23], "1350e": [1, 23], "1012e": [1, 23], "5827e": [1, 23], "9145e": [1, 23], "8987e": [1, 23], "7861e": [1, 23], "4112e": [1, 23], "7514e": [1, 23], "0377e": [1, 23], "6119e": [1, 23], "6974e": [1, 23], "9227e": [1, 23], "5502e": [1, 23], "6419e": [1, 23], "3265e": [1, 23], "1135e": [1, 23], "0431e": [1, 23], "3025e": [1, 23], "0777e": [1, 23], "4116e": [1, 23], "6561e": [1, 23], "2870e": [1, 23], "4176e": [1, 23], "1487e": [1, 23], "5266e": [1, 23], "2469e": [1, 23], "5254e": [1, 23], "2129e": [1, 23], "1837e": [1, 23], "5957e": [1, 23], "3009e": [1, 23], "3448e": [1, 23], "8741e": [1, 23], "7946e": [1, 23], "5803e": [1, 23], "9431e": [1, 23], "3611e": [1, 23], "3890e": [1, 23], "3396e": [1, 23], "8913e": [1, 23], "6739e": [1, 23], "8580e": [1, 23], "4732e": [1, 23], "2845e": [1, 23], "9202e": [1, 23], "6483e": [1, 23], "3382e": [1, 23], "9371e": [1, 23], "8642e": [1, 23], "4136e": [1, 23], "5257e": [1, 23], "5428e": [1, 23], "2954e": [1, 23], "2409e": [1, 23], "3798e": [1, 23], "2413e": [1, 23], "5878e": [1, 23], "6709e": [1, 23], "0508e": [1, 23], "7083e": [1, 23], "0494e": [1, 23], "1418e": [1, 23], "9075e": [1, 23], "6860e": [1, 23], "3186e": [1, 23], "9992e": [1, 23], "1271e": [1, 23], "7508e": [1, 23], "2828e": [1, 23], "0157e": [1, 23], "2795e": [1, 23], "9179e": [1, 23], "6428e": [1, 23], "5829e": [1, 23], "1079e": [1, 23], "41": [1, 3, 23], "4064": [1, 23], "9450": [1, 23], "7715": [1, 23], "1876": [1, 23], "0425": [1, 23], "51": [1, 23], "0653": [1, 23], "volum": [1, 23], "1165": [1, 23], "6177": [1, 23], "medium": [1, 3], "alignn": [1, 3, 5, 6, 11, 18, 23], "gnn": [1, 3, 4, 6, 10, 11, 23], "cfg": [1, 3, 23], "alignnconfig": [1, 3, 4, 6, 10, 23], "alignn_lay": [1, 3, 4, 6, 10, 23], "gcn_layer": [1, 3, 4, 6, 10, 23], "norm": [1, 3, 4, 10, 11, 14], "layernorm": [1, 3, 10, 11, 14], "atom_featur": [1, 3, 4, 6, 10, 11], "embed": [1, 2, 3, 4, 6], "compute_forc": [1, 4, 6, 10, 11, 23], "typeerror": 1, "13": [1, 4, 23], "hostedtoolcach": [1, 4], "python": [1, 4], "x64": [1, 4], "1518": 1, "_wrapped_call_impl": 1, "kwarg": [1, 3], "1516": 1, "_compiled_call_impl": 1, "type": 1, "ignor": 1, "misc": 1, "1517": 1, "_call_impl": 1, "1527": 1, "1522": 1, "If": 1, "ani": [1, 9, 23], "hook": 1, "want": 1, "skip": 1, "rest": 1, "logic": 1, "1523": 1, "function": [1, 3, 4, 6, 12], "just": 1, "forward": [1, 6, 10, 12, 13, 14], "1524": 1, "_backward_hook": 1, "_backward_pre_hook": 1, "_forward_hook": 1, "_forward_pre_hook": 1, "1525": 1, "_global_backward_pre_hook": 1, "_global_backward_hook": 1, "1526": 1, "_global_forward_hook": 1, "_global_forward_pre_hook": 1, "forward_cal": 1, "1529": 1, "try": [1, 3, 4], "1530": 1, "result": [1, 3], "489": 1, "_boundfunct": 1, "__call__": 1, "_": [1, 4], "kw_arg": 1, "488": 1, "_f": 1, "_instanc": 1, "399": 1, "397": 1, "398": 1, "return_typ": 1, "_resolve_method_with_cach": 1, "_convert": 1, "174": 1, "170": 1, "edge_embed": 1, "bondlength": 1, "172": 1, "config": [1, 6, 10, 11], "173": 1, "save": 1, "applic": 1, "edgegatedgraphconv": 1, "edata": [1, 10, 12], "cutoff_valu": 1, "176": 1, "initi": [1, 5], "triplet": 1, "featur": [1, 10, 23], "177": 1, "tupl": [1, 9, 10, 23], "callabl": [1, 9, 10, 11], "util": [1, 3, 5, 6, 22], "dataload": [1, 3, 6, 9], "subsetrandomsampl": [1, 3, 9], "batchsiz": [1, 3], "train_load": [1, 3], "batch_siz": [1, 3, 6], "collate_fn": [1, 3], "collat": [1, 3, 9], "sampler": [1, 3], "split": [1, 3, 4, 9], "drop_last": [1, 3], "next": [1, 3], "iter": [1, 3, 5, 8], "2188": [1, 3], "optim": [1, 3, 4, 6], "an": [1, 3, 5, 6, 23], "explicit": [1, 3], "loop": [1, 3], "see": [1, 3, 5], "quickstart": [1, 3, 22], "tutori": [1, 3], "more": [1, 3], "context": [1, 3], "org": [1, 3, 12, 23], "beginn": [1, 3], "basic": [1, 3, 8], "quickstart_tutori": [1, 3], "html": [1, 3, 12], "tqdm": [1, 3], "criterion": [1, 3, 6], "mseloss": [1, 3, 6], "adamw": [1, 3, 6], "lr": [1, 3, 4, 6], "weight_decai": [1, 3, 6], "training_loss": [1, 3], "epoch": [1, 3, 4, 6], "rang": [1, 3, 4], "step": [1, 3, 4], "enumer": [1, 3], "pred": [1, 3], "loss": [1, 3], "backward": [1, 3], "zero_grad": [1, 3], "19": [1, 3], "65": [1, 3], "09": [1, 3], "91": [1, 3], "24": [1, 3, 21, 23], "42": [1, 3, 6, 9], "matplotlib": [1, 3, 4, 21, 23], "pyplot": [1, 3, 4, 21, 23], "plt": [1, 3, 4, 21, 23], "inlin": [1, 3, 4, 21, 23], "plot": [1, 3, 4, 21, 23], "xlabel": [1, 3, 4], "ylabel": [1, 3, 4], "semilogi": [1, 3, 4], "tempfil": [1, 3], "rank": [1, 3, 9], "training_config": [1, 3], "random_se": [1, 3, 6], "learning_r": [1, 3, 6], "warmup_step": [1, 3, 6], "num_work": [1, 3, 6], "progress": [1, 3, 6], "output_dir": [1, 3, 6], "temporarydirectori": [1, 3], "run_train": [1, 3], "2023": [1, 3], "08": [1, 3, 23], "31": [1, 3], "34": [1, 3], "434": [1, 3], "auto": [1, 3, 6, 23], "auto_dataload": [1, 3], "info": [1, 3], "loader": [1, 3], "collate_default": [1, 3, 9], "0x293eda560": [1, 3], "0x2c9fb2c50": [1, 3], "pin_memori": [1, 3], "0x2c984f790": [1, 3], "start": [1, 3, 10, 23], "avg": [1, 3], "val": [1, 3, 4, 9], "52": [1, 3, 23], "54": [1, 3, 23], "23": [1, 3], "4533848762512207": [1, 3], "l": [1, 3], "ipynb": [1, 3], "test": [1, 3, 9], "properti": [2, 9, 23], "regress": [2, 6], "low": 2, "level": [2, 5, 7], "interfac": [2, 5], "ignit": [2, 6], "trainer": 2, "check": [2, 4], "autograd": 2, "against": 2, "analyt": 2, "tersoff": [2, 18], "potenti": [2, 12, 18], "small": 2, "silicon": 2, "system": [2, 5, 23], "formation_energy_peratom": [3, 9, 23], "3d": [3, 8, 23], "76k": [3, 23], "natur": [3, 6, 23], "articl": [3, 23], "s41524": [3, 23], "020": [3, 23], "00440": [3, 23], "other": [3, 6, 23], "doi": [3, 23], "6084": [3, 23], "m9": [3, 23], "figshar": [3, 4, 23], "6815699": [3, 23], "40": [3, 10, 23], "8m": 3, "0k": 3, "309kib": 3, "200k": 3, "780kib": 3, "913k": 3, "72mib": 3, "71m": 3, "44mib": 3, "87m": 3, "16": [3, 4], "6mib": 3, "30": 3, "1m": 3, "21": 3, "1mib": 3, "4m": 3, "50": [3, 4], "20": 3, "6m": 3, "25": [3, 4], "8mib": 3, "61": 3, "9m": 3, "27": [3, 23], "2mib": 3, "72": 3, "29": [3, 23], "3m": 3, "28": 3, "4mib": 3, "83": 3, "33": [3, 23], "7m": 3, "96": 3, "39": 3, "9mib": 3, "1248": [3, 4], "4276": [3, 23], "project": 3, "poscar": 3, "folder": [3, 6], "bandgap": 3, "unboundlocalerror": 3, "129": 3, "__init__": [3, 4], "df": [3, 9, 23], "custom_collate_fn": [3, 9], "custom_prepare_batch_fn": [3, 9], "train_val_se": [3, 9], "id_tag": [3, 9], "group_id": [3, 9], "n_train": [3, 6, 9], "n_val": [3, 6, 9], "energy_unit": [3, 4, 6, 9, 10, 11], "diskcach": [3, 6, 9], "123": 3, "atomist": [3, 5, 10, 23], "124": 3, "125": 3, "panda": [3, 4, 23], "datafram": [3, 4, 9, 23], "jarvi": [3, 4, 6, 23], "db": [3, 4], "126": 3, "128": [3, 11], "pd": [3, 4], "_load_dataset": 3, "131": 3, "example_id": 3, "iloc": 3, "133": 3, "default": 3, "process": 3, "record": 3, "id": [3, 9], "73": 3, "cache_dir": 3, "70": 3, "elif": 3, "is_dir": 3, "71": 3, "_load_dataset_directori": 3, "local": 3, "variabl": [3, 23], "referenc": 3, "befor": 3, "assign": 3, "0467": 3, "grad_fn": [3, 23], "squeezebackward0": [3, 23], "load_ext": 4, "autoreload": 4, "scipi": 4, "stat": 4, "functool": [4, 6], "partial": [4, 6], "tkagg": 4, "jdata": 4, "periodic_adaptive_radius_graph": [4, 23], "tfm": [4, 6], "ff": [4, 6], "mpf": [4, 6], "subset": [4, 6], "jsonl": [4, 6], "mp3": 4, "get_default_dtyp": 4, "json": [4, 6], "reference_energi": [4, 10, 11], "estimate_reference_energi": [4, 9], "model_cfg": [4, 6], "alignn_ff": [4, 6], "cutoff_onset": [4, 6], "tfmconfig": 4, "layer": [4, 11, 13, 14, 18], "importerror": 4, "get_ipython": 4, "run_line_mag": 4, "1249": 4, "1244": 4, "1245": 4, "1246": 4, "need": [4, 23], "1247": 4, "doe": 4, "librari": [4, 5, 23], "support": [4, 5, 6], "chosen": 4, "instal": [4, 5], "switch_backend": 4, "1250": 4, "except": 4, "1251": 4, "350": 4, "newbackend": 4, "347": 4, "current_framework": 4, "cbook": 4, "_get_running_interactive_framework": 4, "348": 4, "required_framework": 4, "349": 4, "351": 4, "cannot": 4, "requir": [4, 23], "interact": [4, 23], "352": 4, "framework": 4, "run": [4, 6], "353": 4, "355": 4, "new_figure_manag": 4, "356": 4, "357": 4, "export": 4, "358": 4, "keep": 4, "backcompat": 4, "359": 4, "getattr": 4, "tk": 4, "headless": 4, "ckpt": 4, "checkpoint_200": 4, "pt": 4, "map_loc": 4, "devic": [4, 9], "load_state_dict": 4, "kei": [4, 6, 23], "successfulli": 4, "187416": 4, "187417": 4, "187418": 4, "perf": 4, "et": 4, "ep": 4, "ft": 4, "fp": 4, "pval": 4, "fst": 4, "fsp": 4, "idx": 4, "1000": [4, 21, 23], "p": 4, "_fp": 4, "ndim": 4, "unsqueez": 4, "ptrain": 4, "stack": 4, "vstack": 4, "scatter": 4, "collect": 4, "pathcollect": 4, "0x2bb117f40": 4, "fig": [4, 21, 23], "ax": [4, 21, 23], "subplot": [4, 21, 23], "ncol": 4, "figsiz": [4, 21, 23], "pseudolog10": 4, "asinh": 4, "sca": 4, "color": [4, 21], "k": [4, 12, 16], "linestyl": 4, "alpha": 4, "pseudolog": 4, "f_i": 4, "hat": 4, "_i": 4, "xlim": 4, "ylim": 4, "dco": 4, "cosine_similar": 4, "magt": 4, "dim": 4, "magp": 4, "cmap": 4, "inferno": 4, "loglog": 4, "axvlin": 4, "colorbar": 4, "hist": 4, "densiti": 4, "d_": 4, "co": [4, 12], "tight_layout": 4, "sel": 4, "ecdf_plot": 4, "quantil": 4, "annot": 4, "axi": [4, 21, 23], "_y": 4, "arang": 4, "101": [4, 23], "ecdf": 4, "percentil": 4, "ab": 4, "set_ylim": 4, "q": 4, "v": 4, "03f": 4, "xy": 4, "va": 4, "center": 4, "semilogx": 4, "set_xlim": 4, "set_xlabel": 4, "set_ylabel": 4, "probabl": 4, "legend": 4, "loc": [4, 23], "lower": 4, "right": 4, "pct": 4, "80": [4, 10], "vmed": 4, "text": 4, "sgd": 4, "momentum": 4, "schedul": 4, "lr_schedul": 4, "onecyclelr": 4, "max_lr": 4, "pct_start": 4, "steps_per_epoch": 4, "three_phas": 4, "final_div_factor": 4, "get_last_lr": 4, "line2d": 4, "0x2c5939810": 4, "neural": [5, 6, 18], "field": [5, 6, 22], "learn": [5, 6, 23], "toolkit": 5, "experiment": [5, 23], "intend": 5, "rapid": 5, "research": [5, 23], "machin": 5, "main": 5, "goal": 5, "provid": [5, 6, 23], "easili": 5, "extens": 5, "uniform": 5, "develop": 5, "deploi": 5, "codebas": 5, "fork": 5, "modifi": 5, "enabl": [5, 23], "usabl": 5, "improv": 5, "pip": 5, "primari": [5, 23], "gener": 5, "represent": [5, 8], "easi": [5, 23], "effici": [5, 6, 23], "batch": [5, 6, 8, 9, 23], "variou": [5, 23], "consist": [5, 6], "regardless": [5, 6], "materi": 5, "high": 5, "pleas": 5, "report": 5, "bug": 5, "usnistgov": 5, "email": 5, "brian": 5, "decost": 5, "our": 5, "For": [6, 23], "abl": 6, "oper": 6, "preprocess": 6, "structur": [6, 8, 23], "network": [6, 10, 13, 18], "could": [6, 23], "allow": [6, 23], "asynchron": 6, "pyg": [6, 8], "ghost": 6, "pad": 6, "some": [6, 23], "nfflrmodel": 6, "predict": [6, 23], "task": 6, "mai": 6, "vari": 6, "differ": [6, 23], "singl": 6, "scalar": [6, 9, 23], "classif": 6, "where": [6, 23], "n_atom": 6, "n_spatial_dimens": 6, "multi": 6, "might": 6, "nff": 6, "defin": [6, 23], "simplifi": [6, 23], "common": [6, 18], "workflow": 6, "These": [6, 23], "py_config_runn": 6, "flexibl": [6, 23], "two": [6, 13], "full": 6, "rate": 6, "finder": 6, "experi": 6, "transpar": [6, 23], "launch": 6, "idist": 6, "experiment_dir": 6, "__file__": 6, "parent": 6, "resolv": 6, "get_world_s": 6, "3e": 6, "2000": 6, "sourc": [6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], "specif": 6, "data_sourc": 6, "wrk": 6, "chip": 6, "gap0": 6, "document": [7, 12], "describ": 7, "top": 7, "api": 7, "lattic": [8, 21, 23], "batch_num_atom": 8, "int": [8, 9, 10, 11, 13, 14, 16], "what": 8, "fundament": 8, "matrix": [8, 23], "fraction": [8, 23], "occup": 8, "period": [8, 17], "boundari": 8, "condit": 8, "2d": 8, "foundat": 8, "alignngraphtupl": 8, "attribut": [8, 10, 11], "jid": [9, 23], "bool": [9, 10, 11], "liter": [9, 10, 11, 12, 14], "get_energy_and_forc": 9, "split_dataset_by_id": 9, "indic": 9, "stratifi": 9, "trajectori": 9, "split_dataset": 9, "static": 9, "collate_forcefield": 9, "sampl": 9, "list": [9, 12], "helper": [9, 14], "cross": 9, "concaten": 9, "along": 9, "dimens": 9, "global": 9, "whole": 9, "stess": 9, "prepare_batch_default": 9, "non_block": 9, "send": 9, "collate_default_line_graph": 9, "collate_line_graph_ff": 9, "xplor": [10, 11], "batchnorm": [10, 11, 14], "cgcnn": [10, 11, 23], "edge_input_featur": [10, 11], "triplet_input_featur": 10, "embedding_featur": 10, "hidden_featur": 10, "256": 10, "output_featur": [10, 11], "chain": 10, "altern": 10, "gate": 10, "convolut": 10, "heterograph": 10, "ndata": 10, "lg": 10, "z": 10, "angl": [10, 12], "hyperparamet": [10, 11], "schema": [10, 11], "schnetconfig": 11, "d_model": 11, "param": 12, "paper": 12, "1103": 12, "physrevb": 12, "38": 12, "9902": 12, "doc": [12, 13], "pair_tersoff": 12, "fcut": 12, "style": 12, "smooth": 12, "length": 12, "f_repuls": 12, "pair": 12, "repuls": 12, "f_attract": 12, "attract": 12, "angle_cosin": 12, "angular_term": 12, "zeta": 12, "edg": [12, 23], "sum_": 12, "neq": 12, "textrm": 12, "r_": 12, "ik": 12, "theta": 12, "exp": 12, "lambda_3": 12, "ij": 12, "reli": 12, "requires_grad_": 12, "neighbor": [12, 23], "d_in": 13, "d_hidden": 13, "d_out": 13, "in_featur": 14, "out_featur": 14, "multilay": 14, "perceptron": 14, "linear": 14, "silu": 14, "15": 16, "feedforward": 18, "mlplayer": 18, "periodickshellgraph": 18, "periodicadaptiveradiusgraph": 18, "schnet": 18, "ey": [21, 23], "0000": [21, 23], "5000": [21, 23], "_batch_num_atom": [21, 23], "aseatom": [21, 23], "jmol_color": 21, "visual": [21, 23], "plot_atom": [21, 23], "ase_atom": [21, 23], "scaled_posit": [21, 23], "radii": [21, 23], "rotat": [21, 23], "10x": [21, 23], "20y": [21, 23], "0z": [21, 23], "show_unit_cel": [21, 23], "off": [21, 23], "repres": 23, "wai": 23, "spglib": 23, "store": 23, "facilit": 23, "convers": 23, "deep": 23, "approach": 23, "varieti": 23, "aim": 23, "exploratori": 23, "intern": 23, "nativ": 23, "alignn_model": 23, "no_grad": 23, "warn": 23, "103": 23, "avail": 23, "102": 23, "8231": 23, "multipl": 23, "dispatch": 23, "caus": 23, "its": 23, "neg": 23, "cartesian": 23, "8518": 23, "1723e": 23, "5204e": 23, "5018e": 23, "6858e": 23, "5577e": 23, "mulbackward0": 23, "3810e": 23, "3447e": 23, "2815e": 23, "8976e": 23, "1537e": 23, "2890e": 23, "2096e": 23, "segmentreducebackward": 23, "precomput": 23, "cach": 23, "dure": 23, "node": 23, "8951": 23, "7684e": 23, "9206e": 23, "2780e": 23, "0994e": 23, "2781e": 23, "0251e": 23, "3113e": 23, "4305e": 23, "6689e": 23, "make": 23, "instanc": 23, "conveni": 23, "yield": 23, "5669": 23, "3971": 23, "7500": 23, "7849": 23, "2500": 23, "2151": 23, "3075": 23, "6925": 23, "valid": 23, "contain": 23, "larg": 23, "includ": 23, "non": 23, "quantiti": 23, "selected_col": 23, "formula": 23, "optb88vdw_bandgap": 23, "elastic_tensor": 23, "head": 23, "jvasp": 23, "90856": 23, "ticusia": 23, "42762": 23, "000": 23, "na": 23, "86097": 23, "dyb6": 23, "41596": 23, "64906": 23, "be2osru": 23, "04847": 23, "98225": 23, "kbi": 23, "44140": 23, "472": 23, "vse2": 23, "71026": 23, "136": 23, "chang": 23, "column": 23, "miss": 23, "handl": 23, "manual": 23, "4000": 23, "8000": 23, "7000": 23, "3000": 23, "alignn_ff_db": 23, "m3gnet": 23, "special": 23, "thei": 23, "nameerror": 23}, "objects": {"": [[7, 0, 0, "-", "nfflr"]], "nfflr": [[8, 1, 1, "", "Atoms"], [9, 1, 1, "", "AtomsDataset"], [19, 0, 0, "-", "models"], [20, 0, 0, "-", "nn"]], "nfflr.AtomsDataset": [[9, 2, 1, "", "collate_default"], [9, 2, 1, "", "collate_default_line_graph"], [9, 2, 1, "", "collate_forcefield"], [9, 2, 1, "", "collate_line_graph_ff"], [9, 2, 1, "", "prepare_batch_default"], [9, 2, 1, "", "split_dataset"], [9, 2, 1, "", "split_dataset_by_id"]], "nfflr.models": [[10, 1, 1, "", "ALIGNN"], [10, 1, 1, "", "ALIGNNConfig"], [11, 1, 1, "", "SchNet"], [11, 1, 1, "", "SchNetConfig"], [12, 1, 1, "", "Tersoff"], [12, 1, 1, "", "TersoffConfig"]], "nfflr.models.ALIGNN": [[10, 2, 1, "id0", "forward"]], "nfflr.models.Tersoff": [[12, 2, 1, "", "angular_term"], [12, 2, 1, "", "f_attractive"], [12, 2, 1, "", "f_repulsive"], [12, 2, 1, "", "fcut"], [12, 2, 1, "", "forward"], [12, 2, 1, "", "g"]], "nfflr.nn": [[13, 1, 1, "", "FeedForward"], [14, 1, 1, "", "MLPLayer"], [15, 1, 1, "", "PeriodicAdaptiveRadiusGraph"], [16, 1, 1, "", "PeriodicKShellGraph"], [17, 1, 1, "", "PeriodicRadiusGraph"]], "nfflr.nn.FeedForward": [[13, 2, 1, "", "forward"]], "nfflr.nn.MLPLayer": [[14, 2, 1, "", "forward"]]}, "objtypes": {"0": "py:module", "1": "py:class", "2": "py:method"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "class", "Python class"], "2": ["py", "method", "Python method"]}, "titleterms": {"check": 0, "autograd": 0, "forc": [0, 23], "against": 0, "analyt": 0, "embed": 0, "atom": [0, 8, 21, 23], "model": [0, 6, 19, 23], "tersoff": [0, 12], "potenti": [0, 19], "small": 0, "silicon": 0, "system": 0, "properti": [1, 3], "regress": [1, 3], "exampl": [1, 3, 4], "low": [1, 3], "level": [1, 3], "interfac": [1, 3, 6, 23], "us": [1, 3], "ignit": [1, 3], "base": [1, 3], "nfflr": [1, 3, 5, 7, 15, 16, 17, 19, 20], "trainer": [1, 3], "how": 2, "guid": 2, "an": 4, "load": 4, "from": 4, "checkpoint": 4, "document": 5, "project": 5, "histori": 5, "correspond": 5, "code": 5, "conduct": 5, "overview": 6, "input": [6, 23], "represent": [6, 23], "output": 6, "train": [6, 23], "script": 6, "command": 6, "atomsdataset": [9, 23], "alignn": 10, "schnet": 11, "feedforward": 13, "mlplayer": 14, "nn": [15, 16, 17, 20], "periodicadaptiveradiusgraph": 15, "periodickshellgraph": 16, "periodicradiusgraph": 17, "refer": 18, "graph": [19, 20], "neural": 19, "network": 19, "classic": 19, "interatom": 19, "common": [20, 23], "layer": 20, "construct": 20, "tutori": 22, "quickstart": 23, "data": 23, "field": 23, "util": 23, "dataset": 23}, "envversion": {"sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx.ext.viewcode": 1, "sphinx": 60}, "alltitles": {"Checking autograd forces against analytical forces for Embedded Atom model": [[0, "checking-autograd-forces-against-analytical-forces-for-embedded-atom-model"]], "Tersoff potential": [[0, "tersoff-potential"]], "Small silicon system": [[0, "small-silicon-system"]], "Property regression example": [[1, "property-regression-example"], [3, "property-regression-example"]], "low level interface": [[1, "low-level-interface"], [3, "low-level-interface"]], "using the ignite-based NFFLr trainer": [[1, "using-the-ignite-based-nfflr-trainer"], [3, "using-the-ignite-based-nfflr-trainer"]], "How-to guides": [[2, "how-to-guides"]], "an example": [[4, "an-example"]], "loading from a checkpoint": [[4, "loading-from-a-checkpoint"]], "NFFLr documentation": [[5, "nfflr-documentation"]], "Project history": [[5, null]], "Correspondence": [[5, "correspondence"]], "Code of conduct": [[5, "code-of-conduct"]], "Overview": [[6, "overview"]], "input representation": [[6, "input-representation"]], "modeling interface": [[6, "modeling-interface"]], "output representation": [[6, "output-representation"]], "Training scripts": [[6, "training-scripts"]], "training command": [[6, "training-command"]], "nfflr": [[7, "nfflr"]], "Atoms": [[8, "atoms"], [21, "atoms"]], "AtomsDataset": [[9, "atomsdataset"], [23, "atomsdataset"]], "ALIGNN": [[10, "alignn"]], "SchNet": [[11, "schnet"]], "Tersoff": [[12, "tersoff"]], "FeedForward": [[13, "feedforward"]], "MLPLayer": [[14, "mlplayer"]], "nfflr.nn.PeriodicAdaptiveRadiusGraph": [[15, "nfflr-nn-periodicadaptiveradiusgraph"]], "nfflr.nn.PeriodicKShellGraph": [[16, "nfflr-nn-periodickshellgraph"]], "nfflr.nn.PeriodicRadiusGraph": [[17, "nfflr-nn-periodicradiusgraph"]], "Reference": [[18, "reference"]], "nfflr.models": [[19, "module-nfflr.models"]], "Graph neural networks": [[19, "graph-neural-networks"]], "Classical interatomic potentials": [[19, "classical-interatomic-potentials"]], "nfflr.nn": [[20, "module-nfflr.nn"]], "Common layers": [[20, "common-layers"]], "Graph construction": [[20, "graph-construction"]], "Tutorials": [[22, "tutorials"]], "Quickstart": [[23, "quickstart"]], "Atoms Data": [[23, "atoms-data"]], "Models": [[23, "models"]], "Common model interface": [[23, "common-model-interface"]], "Force field models": [[23, "force-field-models"]], "input representations": [[23, "input-representations"]], "Training utilities": [[23, "training-utilities"]], "Force field datasets": [[23, "force-field-datasets"]]}, "indexentries": {"module": [[7, "module-nfflr"], [19, "module-nfflr.models"], [20, "module-nfflr.nn"]], "nfflr": [[7, "module-nfflr"]], "atoms (class in nfflr)": [[8, "nfflr.Atoms"]], "atomsdataset (class in nfflr)": [[9, "nfflr.AtomsDataset"]], "collate_default() (nfflr.atomsdataset static method)": [[9, "nfflr.AtomsDataset.collate_default"]], "collate_default_line_graph() (nfflr.atomsdataset static method)": [[9, "nfflr.AtomsDataset.collate_default_line_graph"]], "collate_forcefield() (nfflr.atomsdataset static method)": [[9, "nfflr.AtomsDataset.collate_forcefield"]], "collate_line_graph_ff() (nfflr.atomsdataset static method)": [[9, "nfflr.AtomsDataset.collate_line_graph_ff"]], "prepare_batch_default() (nfflr.atomsdataset static method)": [[9, "nfflr.AtomsDataset.prepare_batch_default"]], "split_dataset() (nfflr.atomsdataset method)": [[9, "nfflr.AtomsDataset.split_dataset"]], "split_dataset_by_id() (nfflr.atomsdataset method)": [[9, "nfflr.AtomsDataset.split_dataset_by_id"]], "alignn (class in nfflr.models)": [[10, "nfflr.models.ALIGNN"]], "alignn.forward() (in module nfflr.models)": [[10, "nfflr.models.ALIGNN.forward"]], "alignnconfig (class in nfflr.models)": [[10, "nfflr.models.ALIGNNConfig"]], "forward() (nfflr.models.alignn method)": [[10, "id0"]], "schnet (class in nfflr.models)": [[11, "nfflr.models.SchNet"]], "schnetconfig (class in nfflr.models)": [[11, "nfflr.models.SchNetConfig"]], "tersoff (class in nfflr.models)": [[12, "nfflr.models.Tersoff"]], "tersoffconfig (class in nfflr.models)": [[12, "nfflr.models.TersoffConfig"]], "angular_term() (nfflr.models.tersoff method)": [[12, "nfflr.models.Tersoff.angular_term"]], "f_attractive() (nfflr.models.tersoff method)": [[12, "nfflr.models.Tersoff.f_attractive"]], "f_repulsive() (nfflr.models.tersoff method)": [[12, "nfflr.models.Tersoff.f_repulsive"]], "fcut() (nfflr.models.tersoff method)": [[12, "nfflr.models.Tersoff.fcut"]], "forward() (nfflr.models.tersoff method)": [[12, "nfflr.models.Tersoff.forward"]], "g() (nfflr.models.tersoff method)": [[12, "nfflr.models.Tersoff.g"]], "feedforward (class in nfflr.nn)": [[13, "nfflr.nn.FeedForward"]], "forward() (nfflr.nn.feedforward method)": [[13, "nfflr.nn.FeedForward.forward"]], "mlplayer (class in nfflr.nn)": [[14, "nfflr.nn.MLPLayer"]], "forward() (nfflr.nn.mlplayer method)": [[14, "nfflr.nn.MLPLayer.forward"]], "periodicadaptiveradiusgraph (class in nfflr.nn)": [[15, "nfflr.nn.PeriodicAdaptiveRadiusGraph"]], "periodickshellgraph (class in nfflr.nn)": [[16, "nfflr.nn.PeriodicKShellGraph"]], "periodicradiusgraph (class in nfflr.nn)": [[17, "nfflr.nn.PeriodicRadiusGraph"]], "nfflr.models": [[19, "module-nfflr.models"]], "nfflr.nn": [[20, "module-nfflr.nn"]]}})