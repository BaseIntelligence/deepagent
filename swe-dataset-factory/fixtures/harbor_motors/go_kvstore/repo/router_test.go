package kvstore

import "testing"

func TestRouterUpsert(t *testing.T) {
	r := NewRouter(NewStore())
	n, err := r.Upsert("k", "v")
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Fatalf("size want 1 got %d", n)
	}
	v, err := r.Fetch("k")
	if err != nil || v != "v" {
		t.Fatalf("fetch got %q err=%v", v, err)
	}
}

func TestRouterRemoveMissing(t *testing.T) {
	r := NewRouter(NewStore())
	if err := r.Remove("missing"); err == nil {
		t.Fatal("expected error for missing key")
	}
}
