import os

import numpy as np
import pytest

import physrisk.data.static.world as wd
from physrisk.data.inventory import EmbeddedInventory
from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.data.zarr_reader import ZarrReader
from physrisk.hazard_models.core_hazards import (
    CoreFloodModels,
    CoreInventorySourcePaths,
)
from physrisk.kernel import calculation
from physrisk.kernel.assets import (

    ThermalPowerGeneratingAsset,
)
from physrisk.kernel.impact import calculate_impacts
from physrisk.kernel.impact_distrib import EmptyImpactDistrib
from physrisk.kernel.vulnerability_model import DictBasedVulnerabilityModels
from physrisk.vulnerability_models.thermal_power_generation_models import (
    ThermalPowerGenerationAqueductWaterRiskModel,
    ThermalPowerGenerationRiverineInundationModel,
)


@pytest.fixture
def vul_models_dict_extra():
    return {
        "historical_1985": [
            ThermalPowerGenerationRiverineInundationModel(),
        ],
    }


@pytest.fixture
def setup_assets_extra(wri_power_plant_assets):
    asset_list = wri_power_plant_assets
    filtered = asset_list.loc[
        asset_list["primary_fuel"].isin(["Coal", "Gas", "Nuclear", "Oil"])
    ]
    filtered = filtered[-60 < filtered["latitude"]]

    longitudes = np.array(filtered["longitude"])
    latitudes = np.array(filtered["latitude"])
    primary_fuels = np.array(
        [
            primary_fuel.replace(" and ", "And").replace(" ", "")
            for primary_fuel in filtered["primary_fuel"]
        ]
    )
    capacities = np.array(filtered["capacity_mw"])

    countries, continents = wd.get_countries_and_continents(
        latitudes=latitudes, longitudes=longitudes
    )

    assets = [
        ThermalPowerGeneratingAsset(
            latitude,
            longitude,
            type=primary_fuel,
            location=country,
            capacity=capacity,
        )
        for latitude, longitude, capacity, primary_fuel, country in zip(
            latitudes,
            longitudes,
            capacities,
            primary_fuels,
            countries,
        )
        if country in ["Spain"]
    ]

    return assets


def test_error(load_credentials, setup_assets_extra, vul_models_dict_extra):
    """Calculate impacts for the vulnerability models from use case id STRESSTEST."""
    assets = setup_assets_extra
    out = []
    empty_impact_count = 0
    asset_subtype_none_count = 0
    empty_impact_scenarios = []
    asset_subtype_none_assets = []
    exception_scenarios = []

    for scenario_year, vulnerability_models in vul_models_dict_extra.items():
        scenario, year = scenario_year.split("_")

        devaccess = {
            "OSC_S3_ACCESS_KEY": os.environ.get("OSC_S3_ACCESS_KEY", None),
            "OSC_S3_SECRET_KEY": os.environ.get("OSC_S3_SECRET_KEY", None),
            "OSC_S3_BUCKET": os.environ.get("OSC_S3_BUCKET", None),
            # "OSC_S3_ENDPOINT": os.environ.get("OSC_S3_ENDPOINT", None),
        }
        get_env = devaccess.get
        reader = ZarrReader(get_env=get_env)

        vulnerability_models = DictBasedVulnerabilityModels(
            {ThermalPowerGeneratingAsset: vulnerability_models}
        )

        # Use TUDelft flood models.
        hazard_model = ZarrHazardModel(
            source_paths=CoreInventorySourcePaths(
                EmbeddedInventory(), flood_model=CoreFloodModels.TUDelft
            ).source_paths(),
            reader=reader,
        )

        try:
            results = calculate_impacts(
                assets,
                hazard_model,
                vulnerability_models,
                scenario=scenario,
                year=int(year),
            )
        except Exception as e:
            exception_scenarios.append((scenario, year, str(e)))
            continue

        cont = 0
        for result, key in zip(results, results.keys()):
            print(cont)
            cont += 1
            impact = results[key][0].impact
            if isinstance(impact, EmptyImpactDistrib):
                impact_mean = None
                hazard_type = None
                empty_impact_count += 1
                empty_impact_scenarios.append((scenario, year, result.asset.location))
            else:
                impact_mean = impact.mean_impact()
                hazard_type = (
                    impact.hazard_type.__name__
                    if impact.hazard_type.__name__ != "type"
                    else "Wind"
                )

            asset_subtype = result.asset.type if hasattr(result.asset, "type") else None
            if asset_subtype is None:
                asset_subtype_none_count += 1
                asset_subtype_none_assets.append(result.asset.location)

            out.append(
                {
                    "asset": type(result.asset).__name__,
                    "type": getattr(result.asset, "type", None),
                    "location": getattr(result.asset, "location", None),
                    "latitude": result.asset.latitude,
                    "longitude": result.asset.longitude,
                    "impact_mean": impact_mean,
                    "hazard_type": hazard_type if hazard_type else "Wind",
                    "scenario": scenario,
                    "year": int(year),
                }
            )
    print(out)
