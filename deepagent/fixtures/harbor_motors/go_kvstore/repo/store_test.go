package kvstore

import "testing"

func TestStoreSetGet(t *testing.T) {
	s := NewStore()
	s.Set("a", "1")
	v, ok := s.Get("a")
	if !ok || v != "1" {
		t.Fatalf("expected 1, got %q ok=%v", v, ok)
	}
}

func TestStoreDelete(t *testing.T) {
	s := NewStore()
	s.Set("a", "1")
	if !s.Delete("a") {
		t.Fatal("delete should return true")
	}
	if s.Size() != 0 {
		t.Fatalf("size want 0 got %d", s.Size())
	}
}
