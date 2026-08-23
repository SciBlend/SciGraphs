import wgpu

# request_adapter_sync + request_device_sync cost ~180 ms together (RTX 4060
# Laptop, driver 580.173.02, Vulkan). Don't hide that behind a lazy property.


class DeviceError(RuntimeError):
    """Base for every failure in this module."""


class NoAdapterError(DeviceError):
    """No GPU this process can reach: no driver, or a container without
    /dev/dri. Retry ``acquire(fallback=True)``."""


class NoDeviceError(DeviceError):
    """An adapter exists and refused a device with these limits. Ask less."""


class DeviceLostError(DeviceError):
    """Driver reset, hang or sleep. Everything made from the device is dead."""


class Context:
    """One adapter, device and queue. They die together, so they are bundled
    here rather than kept as globals."""

    __slots__ = ("adapter", "device", "queue", "_lost")

    def __init__(self, adapter, device):
        self.adapter = adapter
        self.device = device
        self.queue = device.queue
        self._lost = False

    @property
    def info(self):
        """The adapter's identity as a plain dict, for JSON beside a render."""
        raw = dict(self.adapter.info)
        return {
            "vendor": raw.get("vendor", ""),
            "device": raw.get("device", ""),
            "backend": raw.get("backend_type", ""),
            "type": raw.get("adapter_type", ""),
            "driver": raw.get("description", ""),
        }

    def describe(self):
        i = self.info
        return f"{i['backend']} / {i['device']} ({i['type']}, {i['driver']})"

    @property
    def limits(self):
        return dict(self.device.limits)

    def limit(self, name, default=0):
        """One device limit, by its hyphenated wgpu-py name. The WebGPU IDL
        spells them in camelCase; those return ``default`` silently."""
        value = self.device.limits.get(name)
        return default if value is None else value

    def check(self):
        """Raise DeviceLostError if marked dead. With no `device.lost` future
        in this wgpu-py, loss is reported and never detected; a hang passes."""
        if self._lost:
            raise DeviceLostError(
                "device was marked lost; build a new Context and re-upload")

    def mark_lost(self, reason=""):
        self._lost = True
        return DeviceLostError(reason or "device lost")

    def destroy(self):
        try:
            self.device.destroy()
        except Exception:       # noqa: BLE001 - destroy is best-effort
            pass
        self._lost = True


def acquire(power_preference="high-performance", fallback=False,
            required_features=(), required_limits=None):
    """Return a Context, or raise a DeviceError. ``fallback`` asks for a
    software adapter, which a CI box without a GPU needs. It is off by default:
    landing on llvmpipe silently costs about 40x, which reads as a performance
    regression rather than a configuration error."""
    try:
        adapter = wgpu.gpu.request_adapter_sync(
            power_preference=power_preference,
            force_fallback_adapter=bool(fallback))
    except Exception as exc:    # noqa: BLE001 - wgpu raises several types here
        raise NoAdapterError(
            f"no {'fallback ' if fallback else ''}adapter: {exc}") from exc
    if adapter is None:
        raise NoAdapterError("request_adapter_sync returned None")

    try:
        device = adapter.request_device_sync(
            required_features=list(required_features),
            required_limits=dict(required_limits or {}))
    except Exception as exc:    # noqa: BLE001
        raise NoDeviceError(
            f"adapter {adapter.info.get('device', '?')} refused a device: {exc}"
        ) from exc
    if device is None:
        raise NoDeviceError("request_device_sync returned None")

    return Context(adapter, device)
