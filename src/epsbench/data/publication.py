"""Atomic, no-replace publication bound to one verified directory object."""

from __future__ import annotations

import ctypes
import errno
import os
import platform
import secrets
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any


class StablePublicationError(ValueError):
    """Raised when publication cannot stay bound to one safe directory object."""


PublicationBoundaryHook = Callable[[str, Path], None]
PostPublishCheck = Callable[[], None]


def _no_boundary_hook(_boundary: str, _parent: Path) -> None:
    return


def _no_post_publish_check() -> None:
    return


def _validate_leaf_name(name: str) -> None:
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or ":" in name
        or "\x00" in name
    ):
        raise StablePublicationError("publication target name is not one canonical leaf")


def _is_link_or_reparse(value: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(value.st_mode) or bool(
        getattr(value, "st_file_attributes", 0) & reparse_flag
    )


def _preflight_parent(path: Path) -> tuple[Path, tuple[int, int]]:
    try:
        unresolved = path.absolute()
        parent_stat = unresolved.lstat()
    except OSError as error:
        raise StablePublicationError("publication parent must already exist") from error
    if _is_link_or_reparse(parent_stat) or not stat.S_ISDIR(parent_stat.st_mode):
        raise StablePublicationError("publication parent must be a non-alias directory")
    resolved = unresolved.resolve(strict=True)
    cursor = unresolved
    while True:
        try:
            current = cursor.lstat()
        except OSError as error:
            raise StablePublicationError("publication path cannot be inspected") from error
        if _is_link_or_reparse(current):
            raise StablePublicationError("publication path contains an alias")
        if cursor == cursor.parent:
            break
        cursor = cursor.parent
    return resolved, (parent_stat.st_dev, parent_stat.st_ino)


def _write_all(descriptor: int, payload: bytes) -> None:
    position = 0
    while position < len(payload):
        written = os.write(descriptor, payload[position:])
        if written <= 0:
            raise OSError("publication staging write made no progress")
        position += written


def _read_all(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _posix_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _assert_posix_parent_path(parent: Path, expected: tuple[int, int]) -> None:
    try:
        current = parent.lstat()
    except OSError as error:
        raise StablePublicationError("publication parent changed while in use") from error
    if (
        _is_link_or_reparse(current)
        or not stat.S_ISDIR(current.st_mode)
        or _posix_identity(current) != expected
    ):
        raise StablePublicationError("publication parent changed while in use")


def _posix_link_open_file(source_descriptor: int, parent_descriptor: int, name: str) -> None:
    proc_source = f"/proc/self/fd/{source_descriptor}".encode()
    if not Path(proc_source.decode()).exists():
        raise StablePublicationError(
            "handle-bound Linux publication requires the declared procfs primitive"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    linkat = getattr(libc, "linkat", None)
    if linkat is None:
        raise StablePublicationError("Linux linkat is unavailable; refusing unsafe publication")
    linkat.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    linkat.restype = ctypes.c_int
    at_fdcwd = -100
    at_symlink_follow = 0x400
    if linkat(at_fdcwd, proc_source, parent_descriptor, os.fsencode(name), at_symlink_follow) == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), name)
    if error_number == errno.EXDEV:
        raise StablePublicationError("cross-filesystem publication is unsupported")
    raise OSError(error_number, os.strerror(error_number), name)


def _unlink_posix_if_owned(
    parent_descriptor: int,
    name: str,
    expected: tuple[int, int],
) -> None:
    try:
        current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    if _posix_identity(current) == expected:
        os.unlink(name, dir_fd=parent_descriptor)


def _verify_posix_published(
    parent_descriptor: int,
    name: str,
    expected_identity: tuple[int, int],
    payload: bytes,
) -> None:
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        dir_fd=parent_descriptor,
    )
    try:
        value = os.fstat(descriptor)
        if (
            _posix_identity(value) != expected_identity
            or not stat.S_ISREG(value.st_mode)
            or value.st_nlink != 1
            or value.st_size != len(payload)
            or _read_all(descriptor) != payload
        ):
            raise StablePublicationError("published object differs from owned staging bytes")
    finally:
        os.close(descriptor)


def _atomic_publish_posix(
    parent: Path,
    expected_parent_identity: tuple[int, int],
    name: str,
    payload: bytes,
    boundary_hook: PublicationBoundaryHook,
    post_publish_check: PostPublishCheck,
) -> None:
    required = (
        getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    )
    temporary_flag = getattr(os, "O_TMPFILE", 0)
    if not temporary_flag or not getattr(os, "O_NOFOLLOW", 0):
        raise StablePublicationError(
            "this Unix platform lacks the required no-follow unnamed-file primitive"
        )
    parent_descriptor = os.open(parent, os.O_RDONLY | required)
    staging_descriptor: int | None = None
    published_identity: tuple[int, int] | None = None
    try:
        parent_stat = os.fstat(parent_descriptor)
        parent_identity = _posix_identity(parent_stat)
        if not stat.S_ISDIR(parent_stat.st_mode) or parent_identity != expected_parent_identity:
            raise StablePublicationError("publication parent changed before handle binding")
        _assert_posix_parent_path(parent, parent_identity)

        boundary_hook("before_staging_creation", parent)
        _assert_posix_parent_path(parent, parent_identity)
        try:
            staging_descriptor = os.open(
                ".",
                os.O_RDWR | temporary_flag | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=parent_descriptor,
            )
        except OSError as error:
            if error.errno in {errno.EOPNOTSUPP, errno.EINVAL, errno.ENOSYS, errno.EXDEV}:
                raise StablePublicationError(
                    "publication filesystem lacks safe unnamed-file support"
                ) from error
            raise
        _write_all(staging_descriptor, payload)
        os.fsync(staging_descriptor)
        staging_stat = os.fstat(staging_descriptor)
        published_identity = _posix_identity(staging_stat)
        if (
            not stat.S_ISREG(staging_stat.st_mode)
            or staging_stat.st_nlink != 0
            or staging_stat.st_size != len(payload)
            or staging_stat.st_dev != parent_stat.st_dev
        ):
            raise StablePublicationError("publication staging object is not exclusively owned")

        boundary_hook("before_publication", parent)
        _assert_posix_parent_path(parent, parent_identity)
        _posix_link_open_file(staging_descriptor, parent_descriptor, name)
        os.fsync(parent_descriptor)

        _verify_posix_published(parent_descriptor, name, published_identity, payload)
        post_publish_check()
        _assert_posix_parent_path(parent, parent_identity)
        _verify_posix_published(parent_descriptor, name, published_identity, payload)
    except Exception:
        if published_identity is not None:
            _unlink_posix_if_owned(parent_descriptor, name, published_identity)
            os.fsync(parent_descriptor)
        raise
    finally:
        if staging_descriptor is not None:
            os.close(staging_descriptor)
        os.close(parent_descriptor)


def _windows_api() -> dict[str, Any]:
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll")

    class ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]

    class UnicodeString(ctypes.Structure):
        _fields_ = [
            ("Length", wintypes.USHORT),
            ("MaximumLength", wintypes.USHORT),
            ("Buffer", wintypes.LPWSTR),
        ]

    class ObjectAttributes(ctypes.Structure):
        _fields_ = [
            ("Length", wintypes.ULONG),
            ("RootDirectory", wintypes.HANDLE),
            ("ObjectName", ctypes.POINTER(UnicodeString)),
            ("Attributes", wintypes.ULONG),
            ("SecurityDescriptor", wintypes.LPVOID),
            ("SecurityQualityOfService", wintypes.LPVOID),
        ]

    class IoStatusBlock(ctypes.Structure):
        _fields_ = [("Status", ctypes.c_ssize_t), ("Information", ctypes.c_size_t)]

    class FileDispositionInformation(ctypes.Structure):
        _fields_ = [("DeleteFile", ctypes.c_ubyte)]

    class FileLinkInformation(ctypes.Structure):
        _fields_ = [
            ("ReplaceIfExists", ctypes.c_ubyte),
            ("RootDirectory", wintypes.HANDLE),
            ("FileNameLength", wintypes.DWORD),
            ("FileName", wintypes.WCHAR * 1),
        ]

    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    get_info = kernel32.GetFileInformationByHandle
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(ByHandleFileInformation)]
    get_info.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    write_file = kernel32.WriteFile
    write_file.argtypes = [
        wintypes.HANDLE,
        wintypes.LPCVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    write_file.restype = wintypes.BOOL
    read_file = kernel32.ReadFile
    read_file.argtypes = [
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    read_file.restype = wintypes.BOOL
    flush = kernel32.FlushFileBuffers
    flush.argtypes = [wintypes.HANDLE]
    flush.restype = wintypes.BOOL
    set_info = kernel32.SetFileInformationByHandle
    set_info.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    set_info.restype = wintypes.BOOL
    nt_create = ntdll.NtCreateFile
    nt_create.restype = ctypes.c_long
    nt_set = ntdll.NtSetInformationFile
    nt_set.restype = ctypes.c_long

    return locals()


def _win_identity(info: Any) -> tuple[int, int]:
    return (
        int(info.dwVolumeSerialNumber),
        (int(info.nFileIndexHigh) << 32) | int(info.nFileIndexLow),
    )


def _win_info(api: dict[str, Any], handle: Any) -> Any:
    info = api["ByHandleFileInformation"]()
    if not api["get_info"](handle, ctypes.byref(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    return info


def _close_win(api: dict[str, Any], handle: Any) -> None:
    invalid = ctypes.c_void_p(-1).value
    value = handle.value if hasattr(handle, "value") else handle
    if value not in {None, 0, invalid} and not api["close_handle"](handle):
        raise ctypes.WinError(ctypes.get_last_error())


def _open_win_parent(api: dict[str, Any], parent: Path) -> Any:
    access = 0x0001 | 0x0002 | 0x0080 | 0x00100000
    share = 0x1 | 0x2 | 0x4
    flags = 0x02000000 | 0x00200000
    handle = api["create_file"](str(parent), access, share, None, 3, flags, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    info = _win_info(api, handle)
    if not info.dwFileAttributes & 0x10 or info.dwFileAttributes & 0x400:
        _close_win(api, handle)
        raise StablePublicationError("publication parent handle is an alias or not a directory")
    return handle


def _assert_win_parent_path(api: dict[str, Any], parent: Path, expected: tuple[int, int]) -> None:
    current = _open_win_parent(api, parent)
    try:
        if _win_identity(_win_info(api, current)) != expected:
            raise StablePublicationError("publication parent changed while in use")
    finally:
        _close_win(api, current)


def _nt_open_relative(
    api: dict[str, Any],
    parent_handle: Any,
    name: str,
    *,
    create: bool,
    delete_on_close: bool,
    delete_access: bool = False,
) -> Any:
    buffer = ctypes.create_unicode_buffer(name)
    encoded_length = len(name.encode("utf-16-le"))
    unicode_name = api["UnicodeString"](
        encoded_length,
        encoded_length + 2,
        ctypes.cast(buffer, api["wintypes"].LPWSTR),
    )
    attributes = api["ObjectAttributes"](
        ctypes.sizeof(api["ObjectAttributes"]),
        parent_handle,
        ctypes.pointer(unicode_name),
        0x40 | 0x1000,
        None,
        None,
    )
    io_status = api["IoStatusBlock"]()
    opened = api["wintypes"].HANDLE()
    desired_access = 0x80000000 | 0x0080 | 0x00100000
    if create:
        desired_access |= 0x40000000 | 0x00010000
    if delete_access:
        desired_access |= 0x00010000
    options = 0x40 | 0x20 | 0x00200000
    if delete_on_close:
        options |= 0x1000
    status = api["nt_create"](
        ctypes.byref(opened),
        desired_access,
        ctypes.byref(attributes),
        ctypes.byref(io_status),
        None,
        0x80,
        0x1 | 0x2 | 0x4,
        2 if create else 1,
        options,
        None,
        0,
    )
    if status < 0:
        if status == -1073741771:  # STATUS_OBJECT_NAME_COLLISION
            raise FileExistsError(name)
        raise OSError(f"NtCreateFile failed with NTSTATUS 0x{status & 0xFFFFFFFF:08x}")
    return opened


def _write_win(api: dict[str, Any], handle: Any, payload: bytes) -> None:
    position = 0
    while position < len(payload):
        chunk = payload[position : position + 1024 * 1024]
        buffer = ctypes.create_string_buffer(chunk)
        written = api["wintypes"].DWORD()
        if not api["write_file"](handle, buffer, len(chunk), ctypes.byref(written), None):
            raise ctypes.WinError(ctypes.get_last_error())
        if written.value == 0:
            raise OSError("publication staging write made no progress")
        position += written.value
    if not api["flush"](handle):
        raise ctypes.WinError(ctypes.get_last_error())


def _read_win(api: dict[str, Any], handle: Any) -> bytes:
    chunks: list[bytes] = []
    while True:
        buffer = ctypes.create_string_buffer(1024 * 1024)
        read = api["wintypes"].DWORD()
        if not api["read_file"](handle, buffer, len(buffer), ctypes.byref(read), None):
            raise ctypes.WinError(ctypes.get_last_error())
        if read.value == 0:
            return b"".join(chunks)
        chunks.append(buffer.raw[: read.value])


def _link_win_relative(api: dict[str, Any], source: Any, parent: Any, name: str) -> None:
    encoded = name.encode("utf-16-le")
    offset = api["FileLinkInformation"].FileName.offset
    buffer = ctypes.create_string_buffer(offset + len(encoded))
    header = ctypes.cast(buffer, ctypes.POINTER(api["FileLinkInformation"])).contents
    header.ReplaceIfExists = 0
    header.RootDirectory = parent
    header.FileNameLength = len(encoded)
    ctypes.memmove(ctypes.addressof(buffer) + offset, encoded, len(encoded))
    io_status = api["IoStatusBlock"]()
    status = api["nt_set"](
        source,
        ctypes.byref(io_status),
        buffer,
        len(buffer),
        11,
    )
    if status >= 0:
        return
    if status == -1073741771:  # STATUS_OBJECT_NAME_COLLISION
        raise FileExistsError(name)
    if status in {-1073741623, -1073741437}:  # STATUS_NOT_SAME_DEVICE / unsupported
        raise StablePublicationError("cross-filesystem publication is unsupported")
    raise OSError(f"NtSetInformationFile failed with NTSTATUS 0x{status & 0xFFFFFFFF:08x}")


def _delete_win_handle(api: dict[str, Any], handle: Any) -> None:
    disposition = api["FileDispositionInformation"](1)
    io_status = api["IoStatusBlock"]()
    status = api["nt_set"](
        handle,
        ctypes.byref(io_status),
        ctypes.byref(disposition),
        ctypes.sizeof(disposition),
        13,
    )
    if status < 0:
        raise OSError(f"NtSetInformationFile failed with NTSTATUS 0x{status & 0xFFFFFFFF:08x}")


def _unlink_win_if_owned(
    api: dict[str, Any], parent: Any, name: str, expected: tuple[int, int]
) -> None:
    try:
        handle = _nt_open_relative(
            api, parent, name, create=False, delete_on_close=False, delete_access=True
        )
    except OSError:
        return
    try:
        if _win_identity(_win_info(api, handle)) == expected:
            _delete_win_handle(api, handle)
    finally:
        _close_win(api, handle)


def _verify_win_published(
    api: dict[str, Any],
    parent: Any,
    name: str,
    expected_identity: tuple[int, int],
    payload: bytes,
) -> None:
    destination = _nt_open_relative(api, parent, name, create=False, delete_on_close=False)
    try:
        destination_info = _win_info(api, destination)
        size = (int(destination_info.nFileSizeHigh) << 32) | int(destination_info.nFileSizeLow)
        if (
            _win_identity(destination_info) != expected_identity
            or destination_info.dwFileAttributes & (0x10 | 0x400)
            or destination_info.nNumberOfLinks != 1
            or size != len(payload)
            or _read_win(api, destination) != payload
        ):
            raise StablePublicationError("published object differs from owned staging bytes")
    finally:
        _close_win(api, destination)


def _atomic_publish_windows(
    parent: Path,
    expected_parent_identity: tuple[int, int],
    name: str,
    payload: bytes,
    boundary_hook: PublicationBoundaryHook,
    post_publish_check: PostPublishCheck,
) -> None:
    api = _windows_api()
    parent_handle = _open_win_parent(api, parent)
    staging_handle: Any | None = None
    published_identity: tuple[int, int] | None = None
    try:
        parent_info = _win_info(api, parent_handle)
        parent_identity = _win_identity(parent_info)
        if parent_identity != expected_parent_identity:
            raise StablePublicationError("publication parent changed before handle binding")
        _assert_win_parent_path(api, parent, parent_identity)

        boundary_hook("before_staging_creation", parent)
        _assert_win_parent_path(api, parent, parent_identity)
        staging_name = f".{name}.staging-{secrets.token_hex(16)}"
        staging_handle = _nt_open_relative(
            api,
            parent_handle,
            staging_name,
            create=True,
            delete_on_close=True,
        )
        _write_win(api, staging_handle, payload)
        staging_info = _win_info(api, staging_handle)
        published_identity = _win_identity(staging_info)
        staging_size = (int(staging_info.nFileSizeHigh) << 32) | int(staging_info.nFileSizeLow)
        if (
            staging_info.dwFileAttributes & (0x10 | 0x400)
            or staging_info.nNumberOfLinks != 1
            or staging_size != len(payload)
            or staging_info.dwVolumeSerialNumber != parent_info.dwVolumeSerialNumber
        ):
            raise StablePublicationError("publication staging object is not exclusively owned")

        boundary_hook("before_publication", parent)
        _assert_win_parent_path(api, parent, parent_identity)
        _link_win_relative(api, staging_handle, parent_handle, name)
        _close_win(api, staging_handle)
        staging_handle = None

        _verify_win_published(api, parent_handle, name, published_identity, payload)
        post_publish_check()
        _assert_win_parent_path(api, parent, parent_identity)
        _verify_win_published(api, parent_handle, name, published_identity, payload)
    except Exception:
        if published_identity is not None:
            _unlink_win_if_owned(api, parent_handle, name, published_identity)
        raise
    finally:
        if staging_handle is not None:
            _close_win(api, staging_handle)
        _close_win(api, parent_handle)


def atomic_publish_owned_bytes(
    parent: Path,
    name: str,
    payload: bytes,
    *,
    expected_parent_identity: tuple[int, int],
    boundary_hook: PublicationBoundaryHook = _no_boundary_hook,
    post_publish_check: PostPublishCheck = _no_post_publish_check,
) -> None:
    """Publish bytes atomically without replacement through one stable parent handle."""

    _validate_leaf_name(name)
    resolved_parent, preflight_identity = _preflight_parent(parent)
    if preflight_identity != expected_parent_identity:
        raise StablePublicationError("publication parent changed before stable preflight")
    if os.name == "nt":
        _atomic_publish_windows(
            resolved_parent,
            expected_parent_identity,
            name,
            payload,
            boundary_hook,
            post_publish_check,
        )
    elif os.name == "posix" and platform.system() == "Linux":
        _atomic_publish_posix(
            resolved_parent,
            expected_parent_identity,
            name,
            payload,
            boundary_hook,
            post_publish_check,
        )
    else:
        raise StablePublicationError("stable receipt publication is unsupported on this platform")
