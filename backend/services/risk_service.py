"""Risk profiling + portfolio optimization service stub."""
from __future__ import annotations

from loguru import logger
from motor.motor_asyncio import AsyncIOMotorDatabase

from backend.core.exceptions import ModelNotReadyError
from backend.schemas.risk import (
    PortfolioAllocation,
    PortfolioOptimizationRequest,
    RiskProfileRequest,
    RiskProfileResponse,
)


class RiskService:
    COLLECTION = "risk_profiles"

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db
        self._collection = db[self.COLLECTION]

    async def profile(self, req: RiskProfileRequest) -> RiskProfileResponse:
        """Compute a 0–100 risk score from the questionnaire.

        The first implementation can be a deterministic rule-based scorer
        (age, income, horizon, loss tolerance). Replace later with a learned
        model trained on synthetic user data.
        """
        logger.info(f"Risk profile requested (age={req.age}, horizon={req.time_horizon})")
        # TODO: implement scoring in ml_engine.risk.profiler
        raise ModelNotReadyError(
            "Risk profiler not yet implemented",
            details={"age": req.age, "time_horizon": req.time_horizon},
        )

    async def optimize_portfolio(
        self, req: PortfolioOptimizationRequest
    ) -> PortfolioAllocation:
        """Run Markowitz / Black-Litterman / HRP via PyPortfolioOpt."""
        logger.info(f"Portfolio optimization requested for {req.tickers}")
        # TODO: delegate to ml_engine.risk.portfolio_optimizer
        raise ModelNotReadyError(
            "Portfolio optimizer not yet implemented",
            details={"tickers": req.tickers, "objective": req.objective},
        )
