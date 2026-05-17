"""Risk profiling + portfolio optimization endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from backend.api.deps import RiskSvc
from backend.schemas.common import APIResponse
from backend.schemas.risk import (
    PortfolioAllocation,
    PortfolioOptimizationRequest,
    RiskProfileRequest,
    RiskProfileResponse,
)

router = APIRouter(prefix="/risk", tags=["risk"])


@router.post(
    "/profile",
    response_model=APIResponse[RiskProfileResponse],
    summary="Compute a user's risk profile from questionnaire inputs",
)
async def profile(req: RiskProfileRequest, service: RiskSvc) -> APIResponse[RiskProfileResponse]:
    result = await service.profile(req)
    return APIResponse(data=result)


@router.post(
    "/optimize",
    response_model=APIResponse[PortfolioAllocation],
    summary="Optimize a portfolio for the given objective",
)
async def optimize(
    req: PortfolioOptimizationRequest, service: RiskSvc
) -> APIResponse[PortfolioAllocation]:
    result = await service.optimize_portfolio(req)
    return APIResponse(data=result)
