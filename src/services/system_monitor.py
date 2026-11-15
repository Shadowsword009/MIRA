import platform
import subprocess
import psutil
import urllib.request
import json


def get_system_specs():
    return {
        "operating_system": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "total_ram_gb": round(
            psutil.virtual_memory().total / (1024 ** 3), 2
        )
    }


def get_gpu_info():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,utilization.memory,memory.total,memory.used,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            timeout=2
        )

        if result.returncode != 0 or not result.stdout.strip():
            return None

        values = [value.strip() for value in result.stdout.strip().split(",")]

        def to_float(val, default=0.0):
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        name = values[0]
        core_util = to_float(values[1])
        mem_bus_util = to_float(values[2])
        mem_total = to_float(values[3], default=1.0)
        mem_used = to_float(values[4])
        temp = to_float(values[5])
        power = to_float(values[6]) if len(values) > 6 else 0.0

        vram_pct = round((mem_used / mem_total * 100.0), 1) if mem_total > 0 else 0.0
        effective_util = max(core_util, mem_bus_util)

        return {
            "name": name,
            "utilization": effective_util,
            "core_utilization": core_util,
            "memory_bus_utilization": mem_bus_util,
            "vram_percent": vram_pct,
            "memory_total_mb": mem_total,
            "memory_used_mb": mem_used,
            "temperature": temp,
            "power_watts": power
        }

    except Exception:
        return None


def get_ollama_info():
    base_url = "http://localhost:11434"

    try:
        def fetch(endpoint):
            request = urllib.request.Request(
                f"{base_url}{endpoint}"
            )

            with urllib.request.urlopen(request, timeout=2) as response:
                return json.loads(response.read().decode())

        installed_data = fetch("/api/tags")
        loaded_data = fetch("/api/ps")

        loaded_models = {
            model.get("name"): model
            for model in loaded_data.get("models", [])
        }

        installed_models = []

        for model in installed_data.get("models", []):
            name = model.get("name")
            loaded_model = loaded_models.get(name)

            installed_models.append({
                "name": name,
                "size_gb": round(
                    model.get("size", 0) / (1024 ** 3), 2
                ),
                "parameter_size": model.get(
                    "details", {}
                ).get("parameter_size"),
                "quantization": model.get(
                    "details", {}
                ).get("quantization_level"),
                "loaded": loaded_model is not None,
                "memory_vram_gb": round(
                    loaded_model.get("size_vram", 0) / (1024 ** 3), 2
                ) if loaded_model else 0
            })

        return {
            "connected": True,
            "models": installed_models
        }

    except Exception as error:
        return {
            "connected": False,
            "models": [],
            "error": str(error)
        }


def get_system_load():
    memory = psutil.virtual_memory()

    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_used_gb": round(
            memory.used / (1024 ** 3), 2
        ),
        "ram_percent": memory.percent,
        "disk_percent": psutil.disk_usage("/").percent,
        "gpu": get_gpu_info(),
        "ollama": get_ollama_info()
    }


def get_system_status():
    return {
        "specs": get_system_specs(),
        "load": get_system_load()
    }


def get_model_recommendations(status=None):
    if status is None:
        status = get_system_status()
    specs = status["specs"]
    load = status["load"]

    available_ram = (
        specs["total_ram_gb"] - load["ram_used_gb"]
    )

    recommendations = [
        {
            "name": "qwen2.5:1.5b",
            "description": "Lightweight model for faster local inference",
            "recommended": available_ram >= 2
        },
        {
            "name": "qwen3:4b",
            "description": "More capable reasoning with higher memory requirements",
            "recommended": available_ram >= 5
        }
    ]

    return recommendations
