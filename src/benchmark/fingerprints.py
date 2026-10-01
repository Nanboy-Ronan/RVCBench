"""Generation provenance scoped to reachable runtime code and dependencies."""
import ast
import importlib.metadata
from pathlib import Path

from .artifacts import digest, file_hash
from .registry import _ADVERSARY_REGISTRY


def generation_runtime(root, conf, packages):
    root = Path(root)
    target = _ADVERSARY_REGISTRY[str(conf.vc.mode)][str(conf.vc.model)].split(':')[0]
    pending = [target, 'src.benchmark.backends', 'src.benchmark.artifacts',
               'src.benchmark.runner', 'src.utils.seeding']
    files, external = {}, set()

    def locate(module):
        path = root.joinpath(*module.split('.'))
        return path.with_suffix('.py') if path.with_suffix('.py').is_file() else path / '__init__.py'

    def walk(node):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == 'TYPE_CHECKING':
            return
        yield node
        for child in ast.iter_child_nodes(node):
            yield from walk(child)

    while pending:
        module = pending.pop()
        path = locate(module)
        if not path.is_file() or str(path) in files:
            continue
        files[str(path)] = file_hash(path)
        package = module.split('.') if path.name == '__init__.py' else module.split('.')[:-1]
        for node in walk(ast.parse(path.read_text())):
            imports = []
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                parent = '.'.join(package[:len(package) - node.level + 1]) if node.level else ''
                base = '.'.join(p for p in (parent, node.module) if p)
                imports = [base] + [base + '.' + alias.name for alias in node.names]
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == 'import_module' and node.args and isinstance(node.args[0], ast.Constant)
                  and isinstance(node.args[0].value, str)):
                imports = [node.args[0].value]
            for imported in imports:
                if imported.startswith('src.'):
                    # Evaluation is scheduled separately and has its own fingerprint.
                    if not imported.startswith('src.evaluation'):
                        pending.append(imported)
                else:
                    external.add(imported.split('.')[0])
    worker = conf.adversary.get('worker_script_path')
    if worker and Path(str(worker)).is_file():
        path = Path(str(worker)).resolve()
        files[str(path)] = file_hash(path)
    upstream = conf.adversary.get('code_path')
    if upstream and Path(str(upstream)).is_dir():
        for path in sorted(Path(str(upstream)).rglob('*.py')):
            if '.git' not in path.parts and '__pycache__' not in path.parts:
                files[str(path.resolve())] = file_hash(path)
    distributions = importlib.metadata.packages_distributions()
    requested = {name for module in external for name in distributions.get(module, [])}
    # Include installed transitive requirements (e.g. transformers behind qwen-tts).
    # Checking only the directly imported distribution would permit silent runtime drift.
    from packaging.requirements import Requirement
    queue, visited = list(requested), set()
    while queue:
        name = queue.pop()
        canonical = name.lower().replace('_', '-')
        if canonical in visited:
            continue
        visited.add(canonical)
        try:
            requirements = importlib.metadata.requires(name) or []
        except importlib.metadata.PackageNotFoundError:
            continue
        for raw in requirements:
            requirement = Requirement(raw)
            if requirement.marker is not None and not requirement.marker.evaluate({'extra': ''}):
                continue
            requested.add(requirement.name)
            queue.append(requirement.name)
    normalized = {name.lower().replace('_', '-'): version for name, version in packages.items()}
    versions = {name: normalized.get(name.lower().replace('_', '-')) for name in sorted(requested)}
    source_files = {str(Path(p).relative_to(root)) if Path(p).is_relative_to(root) else p: h for p, h in sorted(files.items())}
    return {'source_files': source_files, 'source_sha256': digest(source_files), 'packages': versions,
            'dependency_scope': 'static_local_imports_and_distribution_dependency_closure_v1',
            'limitations': ['Unresolved dynamic upstream imports remain in full run provenance.',
                           'Separate worker Python environments require independent environment capture.']}
