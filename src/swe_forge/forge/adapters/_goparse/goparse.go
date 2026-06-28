// Command goparse extracts top-level function and method declarations from a
// single Go source file using the standard go/ast parser, emitting them as a
// JSON array on stdout. It is built once and cached by the Go language adapter
// so symbol parsing matches go/ast exactly (including receiver methods).
//
// Exit codes: 0 = parsed (JSON array, possibly empty) on stdout; 1 = the source
// is malformed (a clean parse error on stderr); 2 = a usage/IO error.
package main

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
)

type symbol struct {
	Name      string `json:"name"`
	Kind      string `json:"kind"`
	Receiver  string `json:"receiver,omitempty"`
	StartLine int    `json:"start_line"`
	EndLine   int    `json:"end_line"`
	Signature string `json:"signature"`
}

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: goparse <file.go>")
		os.Exit(2)
	}
	path := os.Args[1]
	src, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "read error:", err)
		os.Exit(2)
	}
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, path, src, parser.SkipObjectResolution)
	if err != nil {
		fmt.Fprintln(os.Stderr, "parse error:", err)
		os.Exit(1)
	}

	syms := []symbol{}
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok {
			continue
		}
		s := symbol{Name: fn.Name.Name, Kind: "function"}
		if fn.Recv != nil && len(fn.Recv.List) > 0 {
			s.Kind = "method"
			s.Receiver = receiverType(fn.Recv.List[0].Type)
		}
		start := fset.Position(fn.Pos())
		end := fset.Position(fn.End())
		s.StartLine = start.Line
		s.EndLine = end.Line
		s.Signature = signature(fn, src)
		syms = append(syms, s)
	}

	out, err := json.Marshal(syms)
	if err != nil {
		fmt.Fprintln(os.Stderr, "encode error:", err)
		os.Exit(2)
	}
	fmt.Println(string(out))
}

// signature returns the declaration header (through the result list, before the
// body) so the caller can pin the public interface.
func signature(fn *ast.FuncDecl, src []byte) string {
	end := fn.End()
	if fn.Body != nil {
		end = fn.Body.Lbrace
	}
	header := string(src[fn.Pos()-1 : end-1])
	out := make([]byte, 0, len(header))
	space := false
	for i := 0; i < len(header); i++ {
		c := header[i]
		if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
			space = true
			continue
		}
		if space && len(out) > 0 {
			out = append(out, ' ')
		}
		space = false
		out = append(out, c)
	}
	return string(out)
}

// receiverType renders a receiver type as it appears in source, e.g. "*Stack"
// or "Stack" or a generic "Tree[T]" reduced to its base name.
func receiverType(e ast.Expr) string {
	switch t := e.(type) {
	case *ast.StarExpr:
		return "*" + receiverType(t.X)
	case *ast.Ident:
		return t.Name
	case *ast.IndexExpr:
		return receiverType(t.X)
	case *ast.IndexListExpr:
		return receiverType(t.X)
	case *ast.SelectorExpr:
		return receiverType(t.Sel)
	}
	return ""
}
