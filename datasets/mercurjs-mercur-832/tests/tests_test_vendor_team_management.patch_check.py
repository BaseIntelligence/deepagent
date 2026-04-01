from pathlib import Path


def test_vendor_api_exports_team_management_hooks_and_routes():
    hooks_index = Path("packages/vendor/src/hooks/api/index.ts")
    route_map = Path("packages/vendor/src/get-route-map.tsx")

    assert hooks_index.exists(), "vendor hooks index should exist"
    assert route_map.exists(), "vendor route map should exist"

    hooks_content = hooks_index.read_text()
    route_content = route_map.read_text()

    assert "./members" in hooks_content, "members api hook should be exported"
    assert "./invites" in hooks_content, "invites api hook should be exported"

    assert "members" in route_content.lower(), "vendor route map should include member management routes"
    assert "invite" in route_content.lower(), "vendor route map should include invite flow routes"
