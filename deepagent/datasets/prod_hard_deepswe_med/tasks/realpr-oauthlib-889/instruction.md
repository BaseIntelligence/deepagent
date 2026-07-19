# Implement the Device Authorization Grant token exchange (RFC 8628 §3.4 & §3.5)

The library already provides the building blocks for the device authorization flow, but it lacks the grant type that handles the token endpoint side of the exchange. We need a `DeviceCodeGrant` that lets an authorization server complete a device code flow: a device polls the token endpoint with its device code and, once the user has authorized, receives access/refresh tokens.

## Expected outcomes

1. Add a `DeviceCodeGrant` grant type under the RFC 8628 package that implements the token request handling described in §3.4 (Device Access Token Request) and §3.5 (Device Access Token Response).
2. The grant must validate incoming token requests that use the device code grant type (`urn:ietf:params:oauth:grant-type:device_code`), including presence of the `device_code` and `client_id` parameters.
3. On a valid, authorized request, issue a bearer token response (access token, and refresh token when applicable) consistent with the existing grant type token response conventions.
4. Support the polling error semantics required by §3.5, returning the appropriate errors (e.g. `authorization_pending`, `slow_down`, `access_denied`, `expired_token`) so devices can poll correctly.
5. Wire the new grant into a pre-configured server so it is usable out of the box, and export it from the appropriate package namespaces so consumers can import it alongside the other OAuth2 grant types.
6. Provide a runnable example demonstrating how to set up and use the device code flow with the new grant.

## Constraints

- Keep the implementation on-spec: do **not** add OpenID Connect / `id_token` behavior to the device flow. RFC 8628 does not define it, so it must remain out of scope here.
- Follow the existing grant type architecture (validators, request validation hooks, token creation via the request validator) so the new grant integrates cleanly with the current endpoints.
- Error types specific to the device flow should live in the RFC 8628 error module and subclass the existing OAuth2 error base so they serialize into standard error responses.
- Ensure the new grant type is discoverable from `oauthlib.oauth2` and the RFC 8628 subpackage exports without breaking existing imports.
- The pre-configured server registration for the device grant must not interfere with the OpenID Connect pre-configured endpoints.

## Implementation notes

- The token request handler needs access to a request validator interface capable of confirming that the device code exists, is associated with the client, has been authorized by the user, and has not expired — returning the right error for each failure state.
- Mirror the structure of the existing pre-configured server setup when adding the device grant so `create_token_response` routes device code requests to the new grant.

IMPORTANT: Please work on this in a new branch from main and commit everything when you are done.
