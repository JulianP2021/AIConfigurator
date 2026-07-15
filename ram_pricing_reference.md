# CPU RAM / SSD / SSD-BW Pricing Reference

This document records how the per-unit prices in
`src/hardware/aws_hardware.json` were derived from public AWS on-demand
pricing so they reflect real cloud component costs as closely as possible.

## 1. CPU RAM

### Isolating the RAM-only price from EC2 High Memory instances

AWS's High Memory (`u-*tb1.*`) instances are purpose-built for large-memory
workloads and make it possible to isolate the cost of RAM because several
sizes share the same CPU, network, storage, and GPU configuration while only
changing the amount of system memory.

#### Why the 56xlarge sizes are *not* a clean RAM comparison

| Instance | vCPUs | RAM (GiB) | Network | EBS optimized | Local disk |
|---|---:|---:|---|---|---|
| u-3tb1.56xlarge | 224 | 3,072 | 50 Gbps | yes | 0 GiB |
| u-6tb1.56xlarge | 224 | 6,144 | 100 Gbps | yes | 0 GiB |

The 6 TB 56xlarge has twice the network bandwidth and twice the EBS throughput,
so a price delta between these two sizes is **not** a pure RAM signal.

#### Clean RAM-only set: 112xlarge / metal sizes

The 112xlarge (and equivalent `.metal`) variants all use the same 448-vCPU
platform, the same 100 Gbps network, no EBS optimization, and no local
instance storage.  The only meaningful difference is RAM.

| Instance | vCPUs | RAM (GiB) | Network | EBS optimized | Local disk | Linux on-demand $/hr | $/GiB/hr |
|---|---:|---:|---|---|---:|---:|---:|
| u-6tb1.112xlarge | 448 | 6,144 | 100 Gbps | no | 0 GiB | $54.60 | $0.00889 |
| u-9tb1.112xlarge | 448 | 9,216 | 100 Gbps | no | 0 GiB | $81.90 | $0.00889 |
| u-12tb1.112xlarge | 448 | 12,288 | 100 Gbps | no | 0 GiB | $109.20 | $0.00889 |
| u-18tb1.112xlarge | 448 | 18,432 | 100 Gbps | no | 0 GiB | $163.80 | $0.00889 |
| u-24tb1.112xlarge | 448 | 24,576 | 100 Gbps | no | 0 GiB | $218.40 | $0.00889 |

Pricing is perfectly linear with memory across this family:

```text
$54.60 / 6,144 GiB = $0.00889 / GiB / hr
$218.40 / 24,576 GiB = $0.00889 / GiB / hr
```

### Adopted RAM unit price

```text
cpu_ram_usd_per_gb_hour = $0.0089 / GiB / hr
```

---

## 2. SSD capacity

### Isolating the SSD-only price from EC2 storage-optimized instances

EC2 `i4i` instances are built from identical 8-vCPU "slices", each with a
fixed 3750 GiB of local NVMe SSD.  Scaling the instance size scales vCPU, RAM,
disk, NVMe bandwidth, network, and EBS bandwidth in exact lockstep, so the
whole family is perfectly linear.

| Instance | vCPUs | RAM (GiB) | SSD (GiB) | NVMe BW (Mbps) | Network | $/hr | $/SSD GiB/hr |
|---|---:|---:|---:|---:|---:|---:|---:|
| i4i.8xlarge | 32 | 256 | 3,750 | 10,000 | 18.75 Gbps | $2.746 | $0.000732 |
| i4i.16xlarge | 64 | 512 | 7,500 | 20,000 | 37.5 Gbps | $5.491 | $0.000732 |
| i4i.24xlarge | 96 | 768 | 11,250 | 30,000 | 56.25 Gbps | $8.2368 | $0.000732 |
| i4i.32xlarge | 128 | 1,024 | 15,000 | 40,000 | 75 Gbps | $10.9824 | $0.000732 |

The $/GiB figure above still contains compute, RAM, and network cost.  To
isolate the **SSD capacity premium** we subtract a compute-only baseline from
the same Ice Lake generation (`c6i` compute-optimized instances, which have no
local disk):

| Instance | vCPUs | RAM (GiB) | Local disk | $/hr | $/vCPU/hr |
|---|---:|---:|---:|---:|---:|
| c6i.8xlarge | 32 | 64 | 0 GiB | $1.36 | $0.04250 |
| c6i.16xlarge | 64 | 128 | 0 GiB | $2.72 | $0.04250 |
| c6i.32xlarge | 128 | 256 | 0 GiB | $5.44 | $0.04250 |

Subtracting `vCPUs * $0.04250` from each `i4i` price leaves a residual that
is almost perfectly linear in SSD capacity:

| Instance | Residual ($/hr) | Residual / SSD GiB |
|---|---:|---:|
| i4i.8xlarge | $1.3860 | $0.000370 |
| i4i.16xlarge | $2.7710 | $0.000369 |
| i4i.24xlarge | $4.1568 | $0.000369 |
| i4i.32xlarge | $5.5424 | $0.000369 |

### Adopted SSD capacity unit price

```text
ssd_usd_per_gb_hour = $0.00037 / GiB / hr
```

---

## 3. SSD bandwidth

The same residual is also linear in NVMe bandwidth, so it can also be priced
as a bandwidth component.  Residual divided by NVMe bandwidth in Gbps:

| Instance | Residual ($/hr) | NVMe BW (Gbps) | Residual / NVMe Gbps/hr |
|---|---:|---:|---:|
| i4i.8xlarge | $1.3860 | 10 | $0.1386 |
| i4i.16xlarge | $2.7710 | 20 | $0.1386 |
| i4i.24xlarge | $4.1568 | 30 | $0.1386 |
| i4i.32xlarge | $5.5424 | 40 | $0.1386 |

Because `i4i` bundles one fixed-size NVMe drive per 8-vCPU slice, the
bandwidth and capacity premiums cannot be separated from this data alone.
For custom hardware we therefore price **both** independently: a machine pays
for its SSD capacity and for its SSD bandwidth.  This lets the simulator
account for systems that have, for example, large but slow SSDs or small but
fast SSDs.

### Adopted SSD bandwidth unit price

```text
ssd_bw_usd_per_gbps_hour = $0.1386 / Gbps / hr
```

---

## 4. Internet / inter-node network bandwidth

### Isolating the network-only premium

The cleanest comparison is between two instances from the same CPU generation
and with the same vCPU/RAM ratio, where the only meaningful difference is the
rated network bandwidth:

| Instance | vCPUs | RAM (GiB) | Network | Linux on-demand $/hr |
|---|---:|---:|---|---:|
| c6g.8xlarge | 32 | 64 | up to 12 Gbps | $1.088 |
| c6gn.8xlarge | 32 | 64 | 50 Gbps | $1.3824 |
| c6gn.16xlarge | 64 | 128 | 100 Gbps | $2.7648 |

The `c6gn` sizes are perfectly 2x linear, so the network premium can be
isolated by subtracting the `c6g` compute/RAM baseline:

| Instance | Baseline (`c6g`) | Residual | Network (Gbps) | Residual / Gbps/hr |
|---|---:|---:|---:|---:|
| c6gn.8xlarge | $1.088 | $0.2944 | 50 | $0.0059 |
| c6gn.16xlarge | $2.176 | $0.5888 | 100 | $0.0059 |

This residual includes the cost of the NIC, EFA, and higher EBS throughput,
so it is an upper-bound on raw network bandwidth.  For the simulator's
inter-node link we use this as a practical cloud-network price.

### Adopted network bandwidth unit price

The simulator's pricing model uses bandwidth prices per **Gbps/hour**, so the
inter-node keys are stored per Gbps rather than per GB transferred:

```text
inter_node_up_usd_per_gbps_hour   = $0.0059 / Gbps / hr
inter_node_down_usd_per_gbps_hour = $0.0059 / Gbps / hr
```

### Estimated cost of a 100 Gbps inter-node link

```text
100 Gbps * $0.0059 / Gbps / hr = $0.59 / hr
```

AWS does not sell bare 100 Gbps cross-connects on EC2, but a `c6gn.16xlarge` is
rated at 100 Gbps and its network premium over the equivalent compute-only
instance is about **$0.59/hour**.  We use that as the working estimate for a
100 Gbps datacenter link in this project.

---

## 5. GPU sanity check: Tesla V100

Using the component prices above, we can back out an implied hourly price for
an NVIDIA Tesla V100 from the EC2 `p3` family.  These instances use the same
Broadwell host generation and scale linearly with GPU count:

| Instance | V100s | vCPUs | RAM (GiB) | Network | Linux on-demand $/hr |
|---|---:|---:|---:|---|---:|
| p3.2xlarge | 1 | 8 | 61 | up to 10 Gbps | $3.06 |
| p3.8xlarge | 4 | 32 | 244 | 10 Gbps | $12.24 |
| p3.16xlarge | 8 | 64 | 488 | 25 Gbps | $24.48 |

Subtracting RAM and inter-node network costs (using the prices from sections 1
and 4) leaves the following GPU-only residual:

| Instance | Non-GPU cost | Residual | Implied V100 $/hr |
|---|---:|---:|---:|
| p3.2xlarge | $0.66 | $2.40 | $2.40 |
| p3.8xlarge | $2.29 | $9.95 | $2.49 |
| p3.16xlarge | $4.49 | $19.99 | $2.50 |

The `p3.2xlarge` estimate is a bit low because its non-GPU costs are small and
rounding errors dominate.  The 4- and 8-GPU sizes agree almost perfectly on a
V100 price of **$2.49 / GPU / hour**.  This matches the AWS "all-in" price of
$3.06/GPU for a single-GPU `p3.2xlarge` once the host/DRAM/network components
are removed.

### Sanity-check against K80 (p2)

The older `p2` family uses Tesla K80s and is not built from fixed-ratio slices,
so the implied K80 price varies ($0.24–$0.48 / GPU / hr).  This confirms that the
`p3` family is a much cleaner source for GPU-only pricing and that the $2.49 V100
estimate is reasonable.

---

## 6. Where these values are used

All unit prices live in `src/hardware/aws_hardware.json` under the
`_pricing` key.  They are consumed by `src/hardware/scraper.py::_derive_custom_price`
to compute `dph_base` for custom hardware entries that do not specify an
explicit hourly price.

| Price key | Value | Source |
|---|---|---:|
| `cpu_ram_usd_per_gb_hour` | $0.0089 / GiB / hr | EC2 `u-*tb1.112xlarge` |
| `ssd_usd_per_gb_hour` | $0.00037 / GiB / hr | EC2 `i4i.*xlarge` residual |
| `ssd_bw_usd_per_gbps_hour` | $0.1386 / Gbps / hr | EC2 `i4i.*xlarge` residual |
| `inter_node_up_usd_per_gbps_hour` | $0.0059 / Gbps / hr | EC2 `c6g` vs `c6gn` residual |
| `inter_node_down_usd_per_gbps_hour` | $0.0059 / Gbps / hr | EC2 `c6g` vs `c6gn` residual |

---

## 7. AWS GPU configs in `aws_hardware.json`

AWS GPU instances do **not** decompose cleanly with the CPU/storage/network unit
prices above.  AWS bundles GPU, network, and SSD together far below the sum of
the component-derived prices (e.g. a `p5en.48xlarge` non-GPU component cost is
already ~$60/hr under these unit prices, leaving almost nothing for the GPU).  To
avoid negative GPU-only residuals, GPU configs store an explicit `dph_base` that
matches the real AWS on-demand price per GPU multiplied by GPU count.  Their
non-GPU fields (RAM, NVMe, bandwidth) still reflect actual AWS instance specs so
the simulation bandwidth/cache model is realistic.

| Config | GPU | GPUs | RAM (GiB) | NVMe (GiB) | NVMe BW (Gbps) | Network (Gbps) | `dph_base` |
|---|---|---:|---:|---:|---:|---:|---:|
| Tesla V100 x1/x4/x8 | Tesla V100 | 1/4/8 | 61/244/488 | 0/0/0 | 0/0/0 | 10/10/25 | $3.06 / $12.24 / $24.48 |
| A100 40GB x8 | A100 40GB | 8 | 1,152 | 8,000 | 80 | 100 | $21.96 |
| A100 80GB x8 | A100 80GB | 8 | 1,152 | 8,000 | 80 | 100 | $27.45 |
| H100 NVL x1/x4/x8 | H100 80GB | 1/4/8 | 256/1,024/2,048 | 3,798/15,192/30,384 | 10/40/80 | 100 | $6.88 / $27.52 / $55.04 |
| H200 x8 / H200 NVL x8 | H200 141GB | 8 | 2,048 | 30,384 | 80 | 100 | $63.30 |

Because AWS does not publish a clean per-GPU residual for A100, H100, or H200,
these `dph_base` values are direct AWS list prices, not component-derived
residuals.  The GPU entries in `src/hardware/_gpu_db.json` keep their normal
compute/memory specifications; only the hourly price is taken from the machine
config.
