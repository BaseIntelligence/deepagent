package kvstore

// Store is an in-memory string map with TTL-aware helpers used by the router.
type Store struct {
	data map[string]string
}

// NewStore creates an empty store.
func NewStore() *Store {
	return &Store{data: make(map[string]string)}
}

// Set writes a key/value pair.
func (s *Store) Set(key, value string) {
	if s.data == nil {
		s.data = make(map[string]string)
	}
	s.data[key] = value
}

// Get returns the value and whether the key exists.
func (s *Store) Get(key string) (string, bool) {
	if s.data == nil {
		return "", false
	}
	v, ok := s.data[key]
	return v, ok
}

// Delete removes a key. Reports whether it existed.
func (s *Store) Delete(key string) bool {
	if s.data == nil {
		return false
	}
	if _, ok := s.data[key]; !ok {
		return false
	}
	delete(s.data, key)
	return true
}

// Size returns the number of keys.
func (s *Store) Size() int {
	return len(s.data)
}
