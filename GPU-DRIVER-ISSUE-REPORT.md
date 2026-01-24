# GPU Driver Issue Report - RTX PRO 6000 Blackwell Nodes

**Date:** January 24, 2026  
**Reported by:** sdodl001  
**Cluster:** Waterfield HPC  

---

## Executive Summary

The **RTX PRO 6000 (Blackwell)** GPUs in partitions `rtxp6000flex-1` through `rtxp6000flex-4` are **non-functional** due to a driver compatibility issue. The currently installed proprietary NVIDIA driver does not support the Blackwell architecture GPUs, which require the NVIDIA open-source kernel modules.

---

## Issue Details

### Symptoms
- `nvidia-smi` returns "No devices found"
- Jobs allocated to RTX PRO 6000 nodes cannot access GPU
- `/dev/nvidia*` device nodes are not created

### Affected Partitions
| Partition | GPU Model | Node Pattern | Status |
|-----------|-----------|--------------|--------|
| rtxp6000flex-1 | RTX PRO 6000 (1 GPU) | wf-g4-standard-48-[1-20] | ❌ **BROKEN** |
| rtxp6000flex-2 | RTX PRO 6000 (2 GPU) | wf-g4-standard-96-[1-12] | ❌ **BROKEN** (presumed) |
| rtxp6000flex-3 | RTX PRO 6000 (4 GPU) | wf-g4-standard-192-[1-8] | ❌ **BROKEN** (presumed) |
| rtxp6000flex-4 | RTX PRO 6000 (8 GPU) | wf-g4-standard-384-[1-4] | ❌ **BROKEN** (presumed) |

### Working Partitions (Confirmed)
| Partition | GPU Model | Status |
|-----------|-----------|--------|
| h100flex-1 | H100 80GB HBM3 | ✅ **WORKING** |
| h100flex-2/4/8 | H100 80GB HBM3 | ✅ **WORKING** (presumed) |

---

## Technical Evidence

### Tested Node: `wf-g4-standard-48-1`

**GPU Hardware (Present):**
```
$ lspci -nn | grep -i nvidia
05:00.0 3D controller [0302]: NVIDIA Corporation GB202GL [RTX PRO 6000 Blackwell Server Edition] [10de:2bb5] (rev a1)
```

**Driver Version:**
```
$ modinfo nvidia | grep version
version:        580.126.09
```

**Kernel Error Messages (dmesg):**
```
NVRM: The NVIDIA GPU 0000:05:00.0 (PCI ID: 10de:2bb5)
      installed in this system requires use of the NVIDIA open kernel modules.
NVRM: GPU 0000:05:00.0: RmInitAdapter failed! (0x22:0x56:884)
NVRM: GPU 0000:05:00.0: rm_init_adapter failed, device minor number 0
```

**Device Nodes:**
```
$ ls /dev/nvidia*
(no output - devices not created)
```

### Working Comparison: `wf-a3-highgpu-1g-1` (H100)

**GPU Hardware:**
```
$ lspci | grep -i nvidia
04:00.0 3D controller: NVIDIA Corporation GH100 [H100 SXM5 80GB] (rev a1)
```

**nvidia-smi Output:**
```
GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-4a21c545-2fd2-adcc-5498-0e3ef0d1d7d5)
```

**Same Driver Version:** 580.126.09 - works on H100 (Hopper), fails on RTX PRO 6000 (Blackwell)

---

## Root Cause Analysis

The **NVIDIA GB202GL (Blackwell architecture)** GPU requires the **NVIDIA open-source kernel modules** (`nvidia-open`), not the proprietary modules currently installed.

From NVIDIA documentation:
> Starting with Blackwell architecture (GB100 series), NVIDIA GPUs require the use of open-source kernel modules. The proprietary driver does not support these GPUs.

The installed driver (580.126.09) includes both proprietary and open modules, but the system is loading the **proprietary** modules instead of the **open** modules.

---

## Recommended Fix

### Option 1: Switch to Open Kernel Modules (Preferred)
```bash
# On each affected node, switch from proprietary to open modules:
sudo dnf remove nvidia-driver
sudo dnf install nvidia-driver-open

# Or if using DKMS:
sudo dnf install nvidia-open-dkms
```

### Option 2: Blacklist Proprietary, Load Open Modules
```bash
# Add to /etc/modprobe.d/nvidia.conf:
blacklist nvidia
alias nvidia nvidia-open
alias nvidia-uvm nvidia-open-uvm
```

### Verification After Fix
```bash
# Check which module is loaded:
modinfo nvidia | grep -E "filename|version"
# Should show: /lib/modules/.../nvidia-open.ko

# Verify GPU access:
nvidia-smi -L
# Should list GPU
```

---

## Additional Issues Found

### B200 Partition Issue
- Job 44 stuck in `CONFIGURING` state for 4+ hours on `wf-a4-highgpu-8g-1`
- B200 partition requires `restricted_gpu` QOS (not documented for users)
- B200 also uses Blackwell architecture - likely has same driver issue

---

## Impact

- **All 44 RTX PRO 6000 nodes** across 4 partitions are effectively offline
- Users cannot access the most cost-effective GPU resources
- Jobs are allocated but fail silently (no GPU access)
- Slurm accounting charges users for GPU time that cannot be used

---

## System Information

| Component | Value |
|-----------|-------|
| Kernel | 5.14.0-611.16.1+2.1.el9_7_ciq.x86_64 |
| OS | Rocky Linux 9.7 (CIQ) |
| Driver | 580.126.09 (proprietary) |
| Required | 580.126.09 (open) or newer |

---

## References

- [NVIDIA Driver Installation Guide - Open Kernel Modules](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html)
- [NVIDIA Open GPU Kernel Modules GitHub](https://github.com/NVIDIA/open-gpu-kernel-modules)
- GPU PCI ID: `10de:2bb5` (GB202GL RTX PRO 6000 Blackwell Server Edition)

---

## Contact

Please contact sdodl001 if additional testing or information is needed.
