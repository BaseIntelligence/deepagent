package kvstore

import "fmt"

// Router composes store operations into higher-level request handlers.
type Router struct {
	store *Store
}

// NewRouter builds a router over the given store (or a fresh one).
func NewRouter(store *Store) *Router {
	if store == nil {
		store = NewStore()
	}
	return &Router{store: store}
}

// Put stores a key if it is non-empty.
func (r *Router) Put(key, value string) error {
	if key == "" {
		return fmt.Errorf("empty key")
	}
	r.store.Set(key, value)
	return nil
}

// Fetch returns the value or an error if missing.
func (r *Router) Fetch(key string) (string, error) {
	v, ok := r.store.Get(key)
	if !ok {
		return "", fmt.Errorf("missing key %q", key)
	}
	return v, nil
}

// Upsert writes and returns the current size.
func (r *Router) Upsert(key, value string) (int, error) {
	if err := r.Put(key, value); err != nil {
		return 0, err
	}
	return r.store.Size(), nil
}

// Remove deletes a key or errors when absent.
func (r *Router) Remove(key string) error {
	if !r.store.Delete(key) {
		return fmt.Errorf("missing key %q", key)
	}
	return nil
}
