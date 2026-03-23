from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules
    if isfile(f) and not f.endswith('__init__.py')
]

# Explicitly import trainers to ensure registration
# This ensures that register_trainer() calls are executed
for module_name in __all__:
    try:
        __import__(f'federatedscope.llm.trainer.{module_name}', fromlist=[module_name])
    except Exception as e:
        # Silently ignore import errors (e.g., missing dependencies)
        pass