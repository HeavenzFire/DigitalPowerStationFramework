function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}
export function createSimulationManager() {
    const config = {
        demandBaseMw: 100,
        demandVolatility: 0.15,
        solarCapacityMw: 80,
        windCapacityMw: 60,
        batteryCapacityMwh: 120,
        batteryMaxChargeMw: 40,
        batteryMaxDischargeMw: 40,
        timestepSeconds: 1
    };
    let batterySocMwh = config.batteryCapacityMwh * 0.5;
    let manualGridSetpointMw = null; // when set, try to match this grid exchange
    const history = [];
    function step(nowMs) {
        const hours = (nowMs / 3600000) % 24;
        // Demand: daily shape + noise
        const dailyShape = 0.6 + 0.4 * Math.sin(((hours - 7) / 24) * 2 * Math.PI) ** 2;
        const noise = 1 + (Math.random() * 2 - 1) * config.demandVolatility;
        const demandMw = config.demandBaseMw * dailyShape * noise;
        // Solar: bell curve mid-day
        const solarIrradiance = Math.max(0, Math.sin(((hours - 6) / 12) * Math.PI));
        const solarMw = config.solarCapacityMw * solarIrradiance * (0.9 + 0.1 * Math.random());
        // Wind: stochastic
        const windMw = config.windCapacityMw * (0.3 + 0.4 * Math.random());
        // Greedy dispatch: use renewables first, then battery, then grid
        const renewableMw = solarMw + windMw;
        // Determine target grid exchange
        let targetGridMw;
        if (manualGridSetpointMw !== null) {
            targetGridMw = manualGridSetpointMw;
        }
        else {
            targetGridMw = 0; // try to be self-sufficient on average
        }
        const netRequiredMw = demandMw - renewableMw - targetGridMw;
        // Battery dispatch to meet netRequiredMw (positive means need discharge)
        const dtHours = config.timestepSeconds / 3600;
        let batteryMw = 0;
        if (netRequiredMw > 0) {
            // Need extra power: discharge up to limit and SOC
            const maxPossibleDischarge = Math.min(config.batteryMaxDischargeMw, batterySocMwh / dtHours);
            batteryMw = clamp(netRequiredMw, 0, maxPossibleDischarge);
        }
        else {
            // Excess power: charge up to limit and headroom
            const headroomMwh = config.batteryCapacityMwh - batterySocMwh;
            const maxPossibleCharge = Math.min(config.batteryMaxChargeMw, headroomMwh / dtHours);
            batteryMw = -clamp(-netRequiredMw, 0, maxPossibleCharge);
        }
        // Update SOC
        batterySocMwh = clamp(batterySocMwh - batteryMw * dtHours, 0, config.batteryCapacityMwh);
        // Compute actual grid exchange after battery
        const residualMw = netRequiredMw - batteryMw;
        let gridMw = targetGridMw + residualMw;
        // Curtail if negative grid exchange (export) beyond renewable surplus
        let curtailedMw = 0;
        if (gridMw < -50) {
            // Limit exports to -50 MW for realism
            curtailedMw = -(50 + gridMw); // positive curtailed amount
            gridMw = -50;
        }
        const point = {
            t: nowMs,
            demandMw,
            solarMw,
            windMw,
            batteryMw,
            socMwh: batterySocMwh,
            gridMw,
            curtailedMw
        };
        history.push(point);
        if (history.length > 3600)
            history.shift();
        return point;
    }
    // Kick off interval loop
    setInterval(() => {
        const now = Date.now();
        step(now);
    }, 1000);
    return {
        getTelemetry() {
            const now = history[history.length - 1] ?? step(Date.now());
            return { now, history };
        },
        handleCommand(command, value) {
            switch (command) {
                case 'set_grid_mw': {
                    if (typeof value !== 'number' || !Number.isFinite(value)) {
                        return { ok: false, error: 'value must be a finite number' };
                    }
                    manualGridSetpointMw = clamp(value, -50, 50);
                    return { ok: true };
                }
                case 'clear_grid_setpoint': {
                    manualGridSetpointMw = null;
                    return { ok: true };
                }
                default:
                    return { ok: false, error: `unknown command: ${command}` };
            }
        }
    };
}
